#!/usr/bin/env python3
"""Prepare a mixed finetuning dataset from font files and glyph image folders."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.font_utils import (
    GlyphRenderer,
    SUPPORTED_FONT_SUFFIXES,
    extract_font_name,
    has_valid_outline,
    load_font,
)
from data_processing.pipeline import create_combined_image, create_test_npz


LOGGER = logging.getLogger("prepare_mixed_finetune_dataset")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_TRAIN_CHARS = 9

# Some scanned glyph JPEGs are slightly truncated but still visually usable.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a training dataset from a mixed folder of fonts and glyph images.",
    )
    parser.add_argument("--input-root", default="train")
    parser.add_argument("--source-font", default=None)
    parser.add_argument("--output-dir", default="data/mixed_finetune_dataset")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--max-train-chars-per-font", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert-image-targets", action="store_true")
    parser.add_argument("--no-invert-image-targets", action="store_false", dest="invert_image_targets")
    parser.set_defaults(invert_image_targets=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_codepoint(codepoint: int) -> str:
    return f"U+{codepoint:04X}"


def filename_for_codepoint(codepoint: int, local_index: int) -> str:
    return f"{local_index:05d}_{format_codepoint(codepoint)}.jpg"


def pick_refs(reference_pool: list[int], current_codepoint: int, seed: int) -> Optional[list[int]]:
    pool = [cp for cp in reference_pool if cp != current_codepoint]
    if len(pool) < 8:
        return None
    rng = random.Random(seed + current_codepoint)
    return rng.sample(pool, 8)


def parse_named_codepoint(text: str) -> Optional[int]:
    stem = text.strip()
    if not stem:
        return None
    if len(stem) == 1:
        return ord(stem)
    upper = stem.upper()
    if upper.startswith("U+"):
        try:
            return int(upper[2:], 16)
        except ValueError:
            return None
    return None


def get_font_codepoints(font) -> set[int]:
    cmap = font.getBestCmap() or {}
    return {cp for cp in cmap.keys() if has_valid_outline(font, cp)}


def scan_input_root(input_root: Path, source_font: Path) -> tuple[list[Path], list[Path]]:
    font_targets: list[Path] = []
    image_targets: list[Path] = []

    for child in sorted(input_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        files = [p for p in child.iterdir() if p.is_file()]
        font_files = [p for p in files if p.suffix in SUPPORTED_FONT_SUFFIXES]
        image_files = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]

        if font_files and image_files:
            raise ValueError(f"Directory mixes font files and image files: {child}")
        if font_files:
            font_targets.extend(sorted(font_files, key=lambda p: p.name.lower()))
            continue
        if image_files:
            image_targets.append(child)

    root_font_files = [
        p for p in input_root.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_FONT_SUFFIXES and p.resolve() != source_font.resolve()
    ]
    font_targets.extend(sorted(root_font_files, key=lambda p: p.name.lower()))

    return font_targets, image_targets


def split_codepoints(
    codepoints: Iterable[int],
    requested_test_count: int,
    max_train_chars: Optional[int],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    population = sorted(set(codepoints))
    if len(population) < 10:
        raise ValueError(
            f"Need at least 10 source-supported characters for train+test, got {len(population)}."
        )

    shuffled = list(population)
    random.Random(seed).shuffle(shuffled)

    max_test_count = max(1, len(shuffled) - MIN_TRAIN_CHARS)
    test_count = min(requested_test_count, max_test_count)
    train_count = len(shuffled) - test_count

    if max_train_chars is not None:
        train_count = min(train_count, max_train_chars)
        if train_count < MIN_TRAIN_CHARS:
            raise ValueError(
                f"max_train_chars_per_font={max_train_chars} is too small; need at least {MIN_TRAIN_CHARS}."
            )

    train_codepoints = sorted(shuffled[:train_count])
    remaining = shuffled[train_count:]
    if not remaining:
        raise ValueError("No unseen characters left for the test split.")
    test_codepoints = sorted(remaining[:test_count])
    ignored_codepoints = sorted(remaining[test_count:])
    return train_codepoints, test_codepoints, ignored_codepoints


def build_ref_grid_from_renderer(
    renderer: GlyphRenderer,
    ref_codepoints: list[int],
    cell_size: int = 128,
) -> Optional[Image.Image]:
    if len(ref_codepoints) != 4:
        raise ValueError("Expected 4 codepoints per reference grid.")
    grid = Image.new("RGB", (256, 256), (255, 255, 255))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    for idx, cp in enumerate(ref_codepoints):
        glyph = renderer.render(cp)
        if glyph is None:
            return None
        glyph = glyph.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(glyph, positions[idx])
    return grid


def load_target_image(path: Path, invert_images: bool, resolution: int) -> Image.Image:
    with Image.open(path) as opened:
        img = opened.convert("RGB")
    if invert_images:
        img = ImageOps.invert(img)
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def validate_target_image(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as opened:
            opened.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return str(exc)
    return None


def build_ref_grid_from_paths(
    glyph_map: dict[int, Path],
    ref_codepoints: list[int],
    invert_images: bool,
    cell_size: int = 128,
) -> Image.Image:
    if len(ref_codepoints) != 4:
        raise ValueError("Expected 4 codepoints per reference grid.")
    grid = Image.new("RGB", (256, 256), (255, 255, 255))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    for idx, cp in enumerate(ref_codepoints):
        glyph = load_target_image(glyph_map[cp], invert_images=invert_images, resolution=cell_size)
        grid.paste(glyph, positions[idx])
    return grid


def write_split_metadata(
    metadata_path: Path,
    *,
    font_name: str,
    font_index: int,
    source_font: Path,
    dataset_type: str,
    target_type: str,
    target_descriptor: str,
    matched_count: int,
    train_count: int,
    test_count: int,
    ignored_count: int,
    extracted_count: int,
    failed_count: int,
    resolution: int,
    characters: list[dict],
) -> None:
    payload = {
        "font_name": font_name,
        "font_index": font_index,
        "source_font": str(source_font),
        "dataset_type": dataset_type,
        "target_type": target_type,
        "target_descriptor": target_descriptor,
        "matched_character_count": matched_count,
        "train_character_count": train_count,
        "test_character_count": test_count,
        "ignored_character_count": ignored_count,
        "extracted_count": extracted_count,
        "failed_count": failed_count,
        "resolution": resolution,
        "image_dimensions": "1024x256",
        "layout": "source(256) + target(256) + refs_grid1(256) + refs_grid2(256)",
        "created_at": datetime.now().isoformat(),
        "characters": characters,
    }
    save_json(metadata_path, payload)


def prepare_font_target(
    *,
    source_font_path: Path,
    source_codepoints: set[int],
    target_font_path: Path,
    font_index: int,
    output_dir: Path,
    resolution: int,
    test_count: int,
    max_train_chars: Optional[int],
    seed: int,
    dry_run: bool,
) -> dict:
    target_font, _ = load_font(str(target_font_path))
    target_codepoints = get_font_codepoints(target_font)
    matched = sorted(source_codepoints & target_codepoints)
    font_name = extract_font_name(target_font, target_font_path)

    train_codepoints, test_codepoints, ignored = split_codepoints(
        matched,
        requested_test_count=test_count,
        max_train_chars=max_train_chars,
        seed=seed + font_index,
    )

    result = {
        "font_index": font_index,
        "font_name": font_name,
        "target_kind": "font_file",
        "target_descriptor": str(target_font_path),
        "matched_count": len(matched),
        "train_count": len(train_codepoints),
        "test_count": len(test_codepoints),
        "ignored_count": len(ignored),
    }
    if dry_run:
        return result

    folder_name = f"{font_index:03d}_{target_font_path.stem}"
    train_out = output_dir / "train" / folder_name
    test_out = output_dir / "test" / folder_name
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    source_renderer = GlyphRenderer(str(source_font_path), resolution)
    target_renderer = GlyphRenderer(str(target_font_path), resolution)

    train_chars: list[dict] = []
    failed_train = 0
    for local_index, cp in enumerate(train_codepoints):
        refs = pick_refs(train_codepoints, cp, seed + font_index)
        if refs is None:
            failed_train += 1
            continue
        source_img = source_renderer.render(cp)
        target_img = target_renderer.render(cp)
        ref_grid_1 = build_ref_grid_from_renderer(target_renderer, refs[:4])
        ref_grid_2 = build_ref_grid_from_renderer(target_renderer, refs[4:])
        if source_img is None or target_img is None or ref_grid_1 is None or ref_grid_2 is None:
            failed_train += 1
            continue
        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = filename_for_codepoint(cp, local_index)
        combined.save(train_out / filename, "JPEG", quality=95)
        train_chars.append(
            {
                "codepoint": format_codepoint(cp),
                "character": chr(cp),
                "filename": filename,
                "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
            }
        )

    test_chars: list[dict] = []
    failed_test = 0
    for local_index, cp in enumerate(test_codepoints):
        refs = random.Random(seed + 1000 + font_index + cp).sample(train_codepoints, 8)
        source_img = source_renderer.render(cp)
        target_img = target_renderer.render(cp)
        ref_grid_1 = build_ref_grid_from_renderer(target_renderer, refs[:4])
        ref_grid_2 = build_ref_grid_from_renderer(target_renderer, refs[4:])
        if source_img is None or target_img is None or ref_grid_1 is None or ref_grid_2 is None:
            failed_test += 1
            continue
        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = filename_for_codepoint(cp, local_index)
        combined.save(test_out / filename, "JPEG", quality=95)
        test_chars.append(
            {
                "codepoint": format_codepoint(cp),
                "character": chr(cp),
                "filename": filename,
                "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
            }
        )

    write_split_metadata(
        train_out / "metadata.json",
        font_name=font_name,
        font_index=font_index,
        source_font=source_font_path,
        dataset_type="train",
        target_type="font_file",
        target_descriptor=str(target_font_path),
        matched_count=len(matched),
        train_count=len(train_codepoints),
        test_count=len(test_codepoints),
        ignored_count=len(ignored),
        extracted_count=len(train_chars),
        failed_count=failed_train,
        resolution=resolution,
        characters=train_chars,
    )
    write_split_metadata(
        test_out / "metadata.json",
        font_name=font_name,
        font_index=font_index,
        source_font=source_font_path,
        dataset_type="test",
        target_type="font_file",
        target_descriptor=str(target_font_path),
        matched_count=len(matched),
        train_count=len(train_codepoints),
        test_count=len(test_codepoints),
        ignored_count=len(ignored),
        extracted_count=len(test_chars),
        failed_count=failed_test,
        resolution=resolution,
        characters=test_chars,
    )

    result.update(
        {
            "train_dir": str(train_out),
            "test_dir": str(test_out),
            "written_train_count": len(train_chars),
            "written_test_count": len(test_chars),
        }
    )
    return result


def collect_image_glyphs(glyph_dir: Path, source_codepoints: set[int]) -> tuple[dict[int, Path], list[int], list[str]]:
    glyph_map: dict[int, Path] = {}
    skipped: list[str] = []

    for path in sorted(glyph_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        cp = parse_named_codepoint(path.stem)
        if cp is None:
            skipped.append(f"{path.name}: unsupported filename")
            continue
        if cp not in source_codepoints:
            skipped.append(f"{path.name}: missing from source font")
            continue
        image_error = validate_target_image(path)
        if image_error is not None:
            skipped.append(f"{path.name}: unreadable image ({image_error})")
            continue
        glyph_map[cp] = path

    return glyph_map, sorted(glyph_map.keys()), skipped


def prepare_image_target(
    *,
    source_font_path: Path,
    source_codepoints: set[int],
    glyph_dir: Path,
    font_index: int,
    output_dir: Path,
    resolution: int,
    test_count: int,
    max_train_chars: Optional[int],
    seed: int,
    invert_images: bool,
    dry_run: bool,
) -> dict:
    glyph_map, matched, skipped_files = collect_image_glyphs(glyph_dir, source_codepoints)
    train_codepoints, test_codepoints, ignored = split_codepoints(
        matched,
        requested_test_count=test_count,
        max_train_chars=max_train_chars,
        seed=seed + font_index,
    )

    result = {
        "font_index": font_index,
        "font_name": glyph_dir.name,
        "target_kind": "image_folder",
        "target_descriptor": str(glyph_dir),
        "matched_count": len(matched),
        "train_count": len(train_codepoints),
        "test_count": len(test_codepoints),
        "ignored_count": len(ignored),
        "skipped_files": skipped_files,
    }
    if dry_run:
        return result

    folder_name = f"{font_index:03d}_{glyph_dir.name}"
    train_out = output_dir / "train" / folder_name
    test_out = output_dir / "test" / folder_name
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    source_renderer = GlyphRenderer(str(source_font_path), resolution)

    train_chars: list[dict] = []
    failed_train = 0
    for local_index, cp in enumerate(train_codepoints):
        refs = pick_refs(train_codepoints, cp, seed + font_index)
        if refs is None:
            failed_train += 1
            continue
        source_img = source_renderer.render(cp)
        if source_img is None:
            failed_train += 1
            continue
        try:
            target_img = load_target_image(glyph_map[cp], invert_images=invert_images, resolution=resolution)
            ref_grid_1 = build_ref_grid_from_paths(glyph_map, refs[:4], invert_images=invert_images)
            ref_grid_2 = build_ref_grid_from_paths(glyph_map, refs[4:], invert_images=invert_images)
        except (OSError, ValueError, UnidentifiedImageError):
            failed_train += 1
            continue
        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = filename_for_codepoint(cp, local_index)
        combined.save(train_out / filename, "JPEG", quality=95)
        train_chars.append(
            {
                "codepoint": format_codepoint(cp),
                "character": chr(cp),
                "filename": filename,
                "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
            }
        )

    test_chars: list[dict] = []
    failed_test = 0
    for local_index, cp in enumerate(test_codepoints):
        refs = random.Random(seed + 1000 + font_index + cp).sample(train_codepoints, 8)
        source_img = source_renderer.render(cp)
        if source_img is None:
            failed_test += 1
            continue
        try:
            target_img = load_target_image(glyph_map[cp], invert_images=invert_images, resolution=resolution)
            ref_grid_1 = build_ref_grid_from_paths(glyph_map, refs[:4], invert_images=invert_images)
            ref_grid_2 = build_ref_grid_from_paths(glyph_map, refs[4:], invert_images=invert_images)
        except (OSError, ValueError, UnidentifiedImageError):
            failed_test += 1
            continue
        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = filename_for_codepoint(cp, local_index)
        combined.save(test_out / filename, "JPEG", quality=95)
        test_chars.append(
            {
                "codepoint": format_codepoint(cp),
                "character": chr(cp),
                "filename": filename,
                "reference_codepoints_1": [format_codepoint(r) for r in refs[:4]],
                "reference_codepoints_2": [format_codepoint(r) for r in refs[4:]],
            }
        )

    write_split_metadata(
        train_out / "metadata.json",
        font_name=glyph_dir.name,
        font_index=font_index,
        source_font=source_font_path,
        dataset_type="train",
        target_type="image_folder",
        target_descriptor=str(glyph_dir),
        matched_count=len(matched),
        train_count=len(train_codepoints),
        test_count=len(test_codepoints),
        ignored_count=len(ignored),
        extracted_count=len(train_chars),
        failed_count=failed_train,
        resolution=resolution,
        characters=train_chars,
    )
    write_split_metadata(
        test_out / "metadata.json",
        font_name=glyph_dir.name,
        font_index=font_index,
        source_font=source_font_path,
        dataset_type="test",
        target_type="image_folder",
        target_descriptor=str(glyph_dir),
        matched_count=len(matched),
        train_count=len(train_codepoints),
        test_count=len(test_codepoints),
        ignored_count=len(ignored),
        extracted_count=len(test_chars),
        failed_count=failed_test,
        resolution=resolution,
        characters=test_chars,
    )

    result.update(
        {
            "train_dir": str(train_out),
            "test_dir": str(test_out),
            "written_train_count": len(train_chars),
            "written_test_count": len(test_chars),
        }
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    source_font = Path(args.source_font).resolve() if args.source_font else (input_root / "FZNewKai.ttf").resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not source_font.is_file():
        raise FileNotFoundError(f"Source font not found: {source_font}")

    source_font_obj, _ = load_font(str(source_font))
    source_codepoints = get_font_codepoints(source_font_obj)

    font_targets, image_targets = scan_input_root(input_root, source_font)
    if not font_targets and not image_targets:
        raise RuntimeError(f"No target fonts or image folders found under {input_root}")

    if not args.dry_run:
        ensure_clean_output(output_dir, force=args.force)

    print(f"Source font: {source_font}")
    print(f"Source-supported character count: {len(source_codepoints)}")
    print(f"Font targets: {len(font_targets)}")
    print(f"Image targets: {len(image_targets)}")
    print(f"Image inversion enabled: {args.invert_image_targets}")

    prepared_fonts: list[dict] = []
    skipped_targets: list[dict] = []
    font_index = 1

    for target_font in font_targets:
        try:
            result = prepare_font_target(
                source_font_path=source_font,
                source_codepoints=source_codepoints,
                target_font_path=target_font,
                font_index=font_index,
                output_dir=output_dir,
                resolution=args.resolution,
                test_count=args.test_count,
                max_train_chars=args.max_train_chars_per_font,
                seed=args.seed,
                dry_run=args.dry_run,
            )
            prepared_fonts.append(result)
            print(
                f"[{font_index:03d}] font {target_font.name}: "
                f"matched={result['matched_count']} train={result['train_count']} test={result['test_count']}"
            )
            font_index += 1
        except Exception as exc:  # noqa: BLE001
            skipped_targets.append(
                {
                    "target_kind": "font_file",
                    "target_descriptor": str(target_font),
                    "reason": str(exc),
                }
            )
            print(f"[skip] font {target_font.name}: {exc}")

    for glyph_dir in image_targets:
        try:
            result = prepare_image_target(
                source_font_path=source_font,
                source_codepoints=source_codepoints,
                glyph_dir=glyph_dir,
                font_index=font_index,
                output_dir=output_dir,
                resolution=args.resolution,
                test_count=args.test_count,
                max_train_chars=args.max_train_chars_per_font,
                seed=args.seed,
                invert_images=args.invert_image_targets,
                dry_run=args.dry_run,
            )
            prepared_fonts.append(result)
            skipped_file_count = len(result.get("skipped_files", []))
            print(
                f"[{font_index:03d}] images {glyph_dir.name}: "
                f"matched={result['matched_count']} train={result['train_count']} "
                f"test={result['test_count']} skipped_files={skipped_file_count}"
            )
            font_index += 1
        except Exception as exc:  # noqa: BLE001
            skipped_targets.append(
                {
                    "target_kind": "image_folder",
                    "target_descriptor": str(glyph_dir),
                    "reason": str(exc),
                }
            )
            print(f"[skip] images {glyph_dir.name}: {exc}")

    if not prepared_fonts:
        raise RuntimeError("No usable targets were prepared.")

    num_fonts = len(prepared_fonts)
    num_chars = max(item["train_count"] for item in prepared_fonts)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_root": str(input_root),
        "source_font": str(source_font),
        "output_dir": str(output_dir),
        "resolution": args.resolution,
        "invert_image_targets": args.invert_image_targets,
        "requested_test_count": args.test_count,
        "max_train_chars_per_font": args.max_train_chars_per_font,
        "num_fonts": num_fonts,
        "num_chars": num_chars,
        "fonts": prepared_fonts,
        "skipped_targets": skipped_targets,
    }

    if args.dry_run:
        print("\nDry run complete.")
        print(f"Usable fonts: {num_fonts}")
        print(f"Recommended --num_fonts: {num_fonts}")
        print(f"Recommended --num_chars: {num_chars}")
        if skipped_targets:
            print(f"Skipped targets: {len(skipped_targets)}")
        return

    test_root = output_dir / "test"
    test_npz_path = output_dir / "test.npz"
    test_result = create_test_npz(test_root, test_npz_path)
    manifest["test_npz"] = str(test_npz_path)
    manifest["test_npz_samples"] = int(test_result["samples"])
    manifest["test_npz_size_mb"] = float(test_result["file_size_mb"])

    save_json(output_dir / "dataset_info.json", manifest)

    print("\nDataset prepared successfully.")
    print(f"Output: {output_dir}")
    print(f"Fonts: {num_fonts}")
    print(f"num_chars: {num_chars}")
    print(f"test.npz samples: {test_result['samples']}")
    print(f"Manifest: {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
