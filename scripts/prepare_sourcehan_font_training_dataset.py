#!/usr/bin/env python3
"""Prepare a full-font finetuning dataset from regional Source Han Serif pairs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _launcher_utils import maybe_reexec_with_repo_python

maybe_reexec_with_repo_python(__file__, REPO_ROOT)

from data_processing.font_utils import (  # noqa: E402
    GlyphRenderer,
    SUPPORTED_FONT_SUFFIXES,
    extract_font_name,
    get_renderable_codepoints,
    load_font,
)
from data_processing.pipeline import create_combined_image, create_test_npz  # noqa: E402


REGION_SOURCE_FONTS = {
    "HK": "SourceHanSerifHK-Light.otf",
    "JP": "SourceHanSerifJP-Light.otf",
    "KR": "SourceHanSerifKR-Light.otf",
    "SC": "SourceHanSerifSC-Light.otf",
    "TC": "SourceHanSerifTC-Light.otf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a full-font finetuning dataset using regional Source Han Serif source fonts.",
    )
    parser.add_argument("--input-root", default="train/用于训练的字体数据")
    parser.add_argument("--output-dir", default="data/sourcehan_font_training_dataset")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--eval-chars-per-font", type=int, default=8)
    parser.add_argument("--split-threshold", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regions", default="", help="Optional comma-separated subset of regions, e.g. SC,TC,JP.")
    parser.add_argument("--limit-fonts-per-region", type=int, default=None, help="Optional debug limit per region.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_codepoint(codepoint: int) -> str:
    return f"U+{codepoint:04X}"


def filename_for_entry(global_index: int, codepoint: int) -> str:
    return f"{global_index:05d}_{format_codepoint(codepoint)}.jpg"


def pick_refs(reference_pool: list[int], current_codepoint: int, seed: int) -> list[int]:
    pool = [cp for cp in reference_pool if cp != current_codepoint]
    if len(pool) < 8:
        raise ValueError(f"Need at least 8 distinct references besides {format_codepoint(current_codepoint)}.")
    return random.Random(seed + current_codepoint).sample(pool, 8)


def build_ref_grid(renderer: GlyphRenderer, ref_codepoints: list[int], cell_size: int = 128):
    from PIL import Image

    grid = Image.new("RGB", (256, 256), (255, 255, 255))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    for idx, cp in enumerate(ref_codepoints):
        glyph = renderer.render(cp)
        if glyph is None:
            return None
        glyph = glyph.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(glyph, positions[idx])
    return grid


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def list_target_fonts(region_dir: Path) -> list[Path]:
    return sorted(
        [path for path in region_dir.iterdir() if path.is_file() and path.suffix in SUPPORTED_FONT_SUFFIXES],
        key=lambda path: path.name.lower(),
    )


def split_into_shards(codepoints: list[int], split_threshold: int) -> list[tuple[int, int, str]]:
    if len(codepoints) > split_threshold:
        midpoint = len(codepoints) // 2
        return [(0, midpoint, "part1"), (midpoint, len(codepoints), "part2")]
    return [(0, len(codepoints), "full")]


def write_composite(
    *,
    source_renderer: GlyphRenderer,
    target_renderer: GlyphRenderer,
    codepoint: int,
    refs: list[int],
    output_path: Path,
) -> bool:
    source_img = source_renderer.render(codepoint)
    target_img = target_renderer.render(codepoint)
    ref_grid_1 = build_ref_grid(target_renderer, refs[:4])
    ref_grid_2 = build_ref_grid(target_renderer, refs[4:])
    if source_img is None or target_img is None or ref_grid_1 is None or ref_grid_2 is None:
        return False
    combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
    combined.save(output_path, "JPEG", quality=95)
    return True


def main() -> None:
    args = parse_args()

    input_root = (REPO_ROOT / args.input_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    train_root = output_dir / "train"
    test_root = output_dir / "test"

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_clean_output(output_dir, force=args.force)

    if args.eval_chars_per_font < 1:
        raise ValueError("--eval-chars-per-font must be >= 1")

    font_entries: list[dict] = []
    font_index = 1
    max_num_chars = 0
    total_train_samples = 0
    total_test_samples = 0

    if args.regions.strip():
        requested_regions = [item.strip().upper() for item in args.regions.split(",") if item.strip()]
    else:
        requested_regions = list(REGION_SOURCE_FONTS.keys())
    region_dirs = [name for name in requested_regions if (input_root / name).is_dir()]
    if not region_dirs:
        raise RuntimeError(f"No regional font directories found under {input_root}")

    print(f"Input root: {input_root}")
    print(f"Regions: {', '.join(region_dirs)}")

    for region in region_dirs:
        region_dir = input_root / region
        source_font_path = input_root / REGION_SOURCE_FONTS[region]
        if not source_font_path.is_file():
            raise FileNotFoundError(f"Missing regional source font for {region}: {source_font_path}")

        source_font, _ = load_font(str(source_font_path))
        source_codepoints = get_renderable_codepoints(source_font, filter_empty=True)
        target_fonts = list_target_fonts(region_dir)
        if args.limit_fonts_per_region is not None:
            target_fonts = target_fonts[:args.limit_fonts_per_region]
        if not target_fonts:
            print(f"[skip] region {region}: no target fonts found")
            continue

        print(f"\n[{region}] source font: {source_font_path.name} ({len(source_codepoints)} renderable chars)")

        for target_font_path in target_fonts:
            target_font, _ = load_font(str(target_font_path))
            target_codepoints = get_renderable_codepoints(target_font, filter_empty=True)
            matched = sorted(source_codepoints & target_codepoints)
            font_name = extract_font_name(target_font, target_font_path)

            if len(matched) < 9:
                print(f"[skip] {region}/{target_font_path.name}: matched={len(matched)} (<9)")
                continue

            max_num_chars = max(max_num_chars, len(matched))
            shard_specs = split_into_shards(matched, args.split_threshold)
            codepoint_to_global_idx = {cp: idx for idx, cp in enumerate(matched)}

            entry = {
                "font_index": font_index,
                "region": region,
                "font_name": font_name,
                "target_font": str(target_font_path),
                "source_font": str(source_font_path),
                "matched_count": len(matched),
                "split_threshold": args.split_threshold,
                "shards": [],
            }

            eval_count = min(args.eval_chars_per_font, len(matched))
            eval_candidates = random.Random(args.seed + font_index).sample(matched, eval_count)
            if not args.dry_run:
                source_renderer = GlyphRenderer(str(source_font_path), args.resolution)
                target_renderer = GlyphRenderer(str(target_font_path), args.resolution)

            for shard_idx, (start, end, shard_label) in enumerate(shard_specs, start=1):
                shard_codepoints = matched[start:end]
                shard_suffix = "" if shard_label == "full" else f"_{shard_label}"
                folder_name = f"{font_index:03d}_{region}_{target_font_path.stem}{shard_suffix}"
                written = 0
                failed = 0
                shard_chars = []

                if not args.dry_run:
                    shard_dir = train_root / folder_name
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    for cp in shard_codepoints:
                        refs = pick_refs(matched, cp, args.seed + font_index)
                        filename = filename_for_entry(codepoint_to_global_idx[cp], cp)
                        ok = write_composite(
                            source_renderer=source_renderer,
                            target_renderer=target_renderer,
                            codepoint=cp,
                            refs=refs,
                            output_path=shard_dir / filename,
                        )
                        if not ok:
                            failed += 1
                            continue
                        written += 1
                        shard_chars.append(
                            {
                                "codepoint": format_codepoint(cp),
                                "character": chr(cp),
                                "filename": filename,
                                "global_char_index": codepoint_to_global_idx[cp],
                                "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                                "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
                            }
                        )
                    save_json(
                        shard_dir / "metadata.json",
                        {
                            "created_at": datetime.now().isoformat(),
                            "dataset_type": "train",
                            "font_index": font_index,
                            "region": region,
                            "font_name": font_name,
                            "target_font": str(target_font_path),
                            "source_font": str(source_font_path),
                            "matched_count": len(matched),
                            "shard_label": shard_label,
                            "shard_index": shard_idx,
                            "shard_count": len(shard_specs),
                            "shard_start": start,
                            "shard_end": end,
                            "written_count": written,
                            "failed_count": failed,
                            "characters": shard_chars,
                        },
                    )
                else:
                    written = len(shard_codepoints)

                total_train_samples += written
                entry["shards"].append(
                    {
                        "folder": folder_name,
                        "shard_label": shard_label,
                        "start": start,
                        "end": end,
                        "planned_count": len(shard_codepoints),
                        "written_count": written,
                        "failed_count": failed,
                    }
                )

            eval_folder_name = f"{font_index:03d}_{region}_{target_font_path.stem}_eval"
            eval_written = 0
            eval_failed = 0
            eval_chars = []
            if not args.dry_run:
                eval_dir = test_root / eval_folder_name
                eval_dir.mkdir(parents=True, exist_ok=True)
                for cp in eval_candidates:
                    refs = pick_refs(matched, cp, args.seed + 100000 + font_index)
                    filename = filename_for_entry(codepoint_to_global_idx[cp], cp)
                    ok = write_composite(
                        source_renderer=source_renderer,
                        target_renderer=target_renderer,
                        codepoint=cp,
                        refs=refs,
                        output_path=eval_dir / filename,
                    )
                    if not ok:
                        eval_failed += 1
                        continue
                    eval_written += 1
                    eval_chars.append(
                        {
                            "codepoint": format_codepoint(cp),
                            "character": chr(cp),
                            "filename": filename,
                            "global_char_index": codepoint_to_global_idx[cp],
                            "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                            "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
                        }
                    )
                save_json(
                    eval_dir / "metadata.json",
                    {
                        "created_at": datetime.now().isoformat(),
                        "dataset_type": "test_seen_preview",
                        "font_index": font_index,
                        "region": region,
                        "font_name": font_name,
                        "target_font": str(target_font_path),
                        "source_font": str(source_font_path),
                        "matched_count": len(matched),
                        "requested_eval_count": eval_count,
                        "written_count": eval_written,
                        "failed_count": eval_failed,
                        "note": "Preview/evaluation subset sampled from training characters so all supported characters remain in train.",
                        "characters": eval_chars,
                    },
                )
            else:
                eval_written = eval_count

            total_test_samples += eval_written
            entry["eval_preview"] = {
                "folder": eval_folder_name,
                "requested_count": eval_count,
                "written_count": eval_written,
                "failed_count": eval_failed,
            }
            font_entries.append(entry)

            shard_msg = " + ".join(str(item["planned_count"]) for item in entry["shards"])
            print(
                f"[{font_index:03d}] {region}/{target_font_path.name}: matched={len(matched)} "
                f"train_shards={shard_msg} eval={eval_written}"
            )
            font_index += 1

    if not font_entries:
        raise RuntimeError("No usable fonts were prepared.")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "resolution": args.resolution,
        "num_fonts": len(font_entries),
        "num_chars": max_num_chars,
        "total_train_samples": total_train_samples,
        "total_test_samples": total_test_samples,
        "eval_chars_per_font": args.eval_chars_per_font,
        "split_threshold": args.split_threshold,
        "fonts": font_entries,
    }

    if args.dry_run:
        print("\nDry run complete.")
        print(f"Usable fonts: {len(font_entries)}")
        print(f"Recommended num_fonts: {len(font_entries)}")
        print(f"Recommended num_chars: {max_num_chars}")
        return

    test_result = create_test_npz(test_root, output_dir / "test.npz")
    manifest["test_npz"] = str(output_dir / "test.npz")
    manifest["test_npz_samples"] = int(test_result["samples"])
    manifest["test_npz_size_mb"] = float(test_result["file_size_mb"])
    save_json(output_dir / "dataset_info.json", manifest)

    print("\nDataset prepared successfully.")
    print(f"Output: {output_dir}")
    print(f"Fonts: {len(font_entries)}")
    print(f"num_chars: {max_num_chars}")
    print(f"Train samples: {total_train_samples}")
    print(f"Test preview samples: {total_test_samples}")
    print(f"NPZ samples: {test_result['samples']}")
    print(f"Manifest: {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
