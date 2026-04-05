#!/usr/bin/env python3
"""Generate a full GB2312 charset for one finetuned style folder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _launcher_utils import maybe_reexec_with_repo_python

maybe_reexec_with_repo_python(__file__, REPO_ROOT)

from data_processing.charsets import get_charset_codepoints
from data_processing.font_utils import GlyphRenderer, has_valid_outline, load_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all GB2312 glyphs for one style folder using a finetuned checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="run/lora_ft_large_content_style_encoder/checkpoint-latest.pth",
    )
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--train-dir", default="data/mixed_finetune_dataset/train")
    parser.add_argument("--source-font", default="")
    parser.add_argument("--target-font", default="")
    parser.add_argument("--style-folder", default="")
    parser.add_argument("--font-index", type=int, default=None)
    parser.add_argument("--charset", default="gb2312")
    parser.add_argument("--output-dir", default="run/generated_gb2312")
    parser.add_argument("--temp-npz", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--sampling-method", default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref-index", type=int, default=0, choices=range(8))
    parser.add_argument("--reference-mode", choices=["fixed", "cycle"], default="fixed")
    parser.add_argument("--pairwise", choices=["src_gen", "target_gen"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_font_codepoints(font) -> set[int]:
    cmap = font.getBestCmap() or {}
    return {cp for cp in cmap.keys() if has_valid_outline(font, cp)}


def extract_ref(img: Image.Image, ref_global_idx: int, ref_size: int = 128) -> Image.Image:
    grid_idx = ref_global_idx // 4
    ref_idx = ref_global_idx % 4
    grid_x = 512 if grid_idx == 0 else 768
    row = ref_idx // 2
    col = ref_idx % 2
    x1 = grid_x + col * 128
    y1 = row * 128
    ref = img.crop((x1, y1, x1 + 128, y1 + 128))
    if ref.size != (ref_size, ref_size):
        ref = ref.resize((ref_size, ref_size), Image.LANCZOS)
    return ref


def parse_font_index(folder_name: str) -> int:
    idx_str = folder_name.split("_", 1)[0]
    if idx_str.startswith("'"):
        idx_str = idx_str[1:]
    return int(idx_str) - 1


def load_dataset_info(dataset_dir: Path) -> dict:
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"dataset_info.json not found under {dataset_dir}")
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_font_identity(value: str) -> str:
    return Path(value).stem.lower()


def resolve_font_entry(dataset_info: dict, target_font: str) -> dict:
    target_name = normalize_font_identity(target_font)
    target_basename = Path(target_font).name.lower()

    matches = []
    for entry in dataset_info.get("fonts", []):
        entry_target_font = str(entry.get("target_font", ""))
        entry_font_name = str(entry.get("font_name", ""))
        candidates = {
            normalize_font_identity(entry_target_font),
            Path(entry_target_font).name.lower(),
            normalize_font_identity(entry_font_name),
            entry_font_name.lower(),
        }
        if target_name in candidates or target_basename in candidates:
            matches.append(entry)

    if not matches:
        raise FileNotFoundError(f"Could not find target font '{target_font}' in dataset_info.json")
    if len(matches) > 1:
        names = ", ".join(str(item.get("target_font", "")) for item in matches[:5])
        raise RuntimeError(
            f"Multiple dataset entries matched target font '{target_font}': {names}. "
            "Please pass a more specific path or use --style-folder."
        )
    return matches[0]


def resolve_from_dataset(
    *,
    dataset_dir: Path,
    target_font: str,
    source_font_arg: str,
) -> tuple[Path, Path, Path, int]:
    dataset_info = load_dataset_info(dataset_dir)
    entry = resolve_font_entry(dataset_info, target_font)
    train_dir = dataset_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found under dataset dir: {train_dir}")

    shards = entry.get("shards") or []
    if not shards:
        raise RuntimeError(f"No shard metadata found for target font '{target_font}'")
    shard_folder = str(shards[0].get("folder", ""))
    if not shard_folder:
        raise RuntimeError(f"First shard for target font '{target_font}' is missing a folder name")
    style_folder = train_dir / shard_folder
    if not style_folder.is_dir():
        raise FileNotFoundError(f"Style folder referenced by dataset_info.json not found: {style_folder}")

    if source_font_arg:
        source_font = Path(source_font_arg)
        if not source_font.is_absolute():
            source_font = (REPO_ROOT / source_font_arg).resolve()
    else:
        source_fonts = entry.get("source_fonts") or []
        if not source_fonts:
            raise RuntimeError(f"No source_fonts recorded for target font '{target_font}'")
        source_font = Path(str(source_fonts[0])).resolve()

    if not source_font.is_file():
        raise FileNotFoundError(f"Source font not found: {source_font}")

    style_font_index = int(entry["font_index"]) - 1
    return train_dir, style_folder, source_font, style_font_index


def resolve_style_folder(train_dir: Path, style_folder: str, font_index: int | None) -> Path:
    if bool(style_folder) == (font_index is not None):
        raise ValueError("Specify exactly one of --style-folder or --font-index.")

    if style_folder:
        folder = Path(style_folder)
        if not folder.is_absolute():
            folder = (train_dir / style_folder).resolve()
        if not folder.is_dir():
            candidates = sorted(
                p for p in train_dir.iterdir()
                if p.is_dir()
                and not p.name.startswith(".")
                and (
                    p.name == style_folder
                    or p.name.split("_", 1)[-1] == style_folder
                    or p.name.endswith(f"_{style_folder}")
                    or f"_{style_folder}_" in p.name
                )
            )
            if not candidates:
                raise FileNotFoundError(
                    f"Style folder not found: {folder}. "
                    f"Try the full folder name such as '005_{style_folder}' or use --font-index."
                )
            if len(candidates) > 1:
                names = ", ".join(p.name for p in candidates)
                raise RuntimeError(
                    f"Multiple style folders matched '{style_folder}': {names}. "
                    "Please pass the full folder name or use --font-index."
                )
            return candidates[0]
        return folder

    candidates = sorted(
        p for p in train_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and parse_font_index(p.name) == font_index
    )
    if not candidates:
        raise FileNotFoundError(f"No style folder found for font index {font_index} under {train_dir}")
    return candidates[0]


def collect_train_images(style_folder: Path) -> list[Path]:
    images = sorted(
        p for p in style_folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"No training images found in {style_folder}")
    return images


def build_style_reference_pool(
    train_images: list[Path],
    ref_index: int,
    reference_mode: str,
) -> list[np.ndarray]:
    if reference_mode == "fixed":
        selected_images = train_images[:1]
    else:
        selected_images = train_images

    refs: list[np.ndarray] = []
    for image_path in selected_images:
        with Image.open(image_path) as opened:
            img = opened.convert("RGB")
            ref = extract_ref(img, ref_index)
            refs.append(np.array(ref).transpose(2, 0, 1))

    if not refs:
        raise RuntimeError(f"Could not extract any reference glyphs from {len(train_images)} images.")
    return refs


def build_inference_npz(
    *,
    train_dir: Path,
    style_folder: Path,
    source_font: Path,
    charset: str,
    ref_index: int,
    reference_mode: str,
    output_path: Path,
    style_font_index_override: int | None = None,
) -> dict:
    source_font_obj, source_font_path = load_font(str(source_font))
    renderable_cps = sorted(get_charset_codepoints(charset) & get_font_codepoints(source_font_obj))
    if not renderable_cps:
        raise RuntimeError(f"No renderable codepoints left after intersecting charset={charset} with {source_font}.")

    train_images = collect_train_images(style_folder)
    style_refs = build_style_reference_pool(train_images, ref_index=ref_index, reference_mode=reference_mode)
    style_font_index = style_font_index_override if style_font_index_override is not None else parse_font_index(style_folder.name)
    source_renderer = GlyphRenderer(str(source_font_path), 256)

    num_samples = len(renderable_cps)
    font_labels = np.full(num_samples, style_font_index, dtype=np.int64)
    char_labels = np.arange(num_samples, dtype=np.int64)
    unicode_labels = np.array(renderable_cps, dtype=np.int64)
    content_images = np.empty((num_samples, 3, 256, 256), dtype=np.uint8)
    style_images = np.empty((num_samples, 3, 128, 128), dtype=np.uint8)

    for i, cp in enumerate(renderable_cps):
        content = source_renderer.render(cp)
        if content is None:
            raise RuntimeError(f"Failed to render source glyph U+{cp:04X} from {source_font_path}")
        content_images[i] = np.array(content).transpose(2, 0, 1)
        style_images[i] = style_refs[0] if reference_mode == "fixed" else style_refs[i % len(style_refs)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        font_labels=font_labels,
        char_labels=char_labels,
        unicode_labels=unicode_labels,
        content_images=content_images,
        style_images=style_images,
        num_original_samples=np.int64(num_samples),
    )

    manifest = {
        "checkpoint": "",
        "train_dir": str(train_dir),
        "style_folder": str(style_folder),
        "style_font_index": style_font_index,
        "source_font": str(source_font_path),
        "charset": charset,
        "ref_index": ref_index,
        "reference_mode": reference_mode,
        "num_samples": num_samples,
        "npz_path": str(output_path),
    }
    return manifest


def main() -> None:
    args = parse_args()

    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    temp_npz = Path(args.temp_npz).resolve() if args.temp_npz else (output_dir / "_gb2312_inference.npz")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    style_font_index_override: int | None = None
    if args.dataset_dir:
        dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
        if not args.target_font:
            raise ValueError("--target-font is required when using --dataset-dir.")
        train_dir, style_folder, source_font, style_font_index_override = resolve_from_dataset(
            dataset_dir=dataset_dir,
            target_font=args.target_font,
            source_font_arg=args.source_font,
        )
    else:
        train_dir = (REPO_ROOT / args.train_dir).resolve()
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Train dir not found: {train_dir}")
        if not args.source_font:
            raise ValueError("--source-font is required when --dataset-dir is not used.")
        source_font = (REPO_ROOT / args.source_font).resolve()
        if not source_font.is_file():
            raise FileNotFoundError(f"Source font not found: {source_font}")
        style_folder = resolve_style_folder(train_dir, args.style_folder, args.font_index)

    manifest = build_inference_npz(
        train_dir=train_dir,
        style_folder=style_folder,
        source_font=source_font,
        charset=args.charset,
        ref_index=args.ref_index,
        reference_mode=args.reference_mode,
        output_path=temp_npz,
        style_font_index_override=style_font_index_override,
    )
    manifest["checkpoint"] = str(checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "_gb2312_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Generation plan:")
    print(f"  checkpoint    = {checkpoint}")
    print(f"  style_folder  = {style_folder}")
    print(f"  font_index    = {manifest['style_font_index']}")
    print(f"  source_font   = {source_font}")
    if args.target_font:
        print(f"  target_font   = {args.target_font}")
    if args.dataset_dir:
        print(f"  dataset_dir   = {(REPO_ROOT / args.dataset_dir).resolve()}")
    print(f"  charset       = {args.charset}")
    print(f"  num_samples   = {manifest['num_samples']}")
    print(f"  ref_index     = {args.ref_index}")
    print(f"  ref_mode      = {args.reference_mode}")
    print(f"  temp_npz      = {temp_npz}")
    print(f"  output_dir    = {output_dir}")
    print(f"  manifest      = {manifest_path}")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "generate_chars.py"),
        "--checkpoint", str(checkpoint),
        "--test_npz", str(temp_npz),
        "--output_dir", str(output_dir),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.cfg is not None:
        cmd.extend(["--cfg", str(args.cfg)])
    if args.sampling_method is not None:
        cmd.extend(["--sampling_method", args.sampling_method])
    if args.num_sampling_steps is not None:
        cmd.extend(["--num_sampling_steps", str(args.num_sampling_steps)])
    if args.pairwise is not None:
        cmd.extend(["--pairwise", args.pairwise])

    print("\nGeneration command:")
    print(" ".join(str(part) for part in cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
