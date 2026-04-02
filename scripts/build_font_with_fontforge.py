#!/usr/bin/env python3
"""Build a TTF from generated glyph PNGs using potrace + FontForge."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _launcher_utils import maybe_reexec_with_repo_python

maybe_reexec_with_repo_python(__file__, REPO_ROOT)

from data_processing.font_utils import is_cjk_codepoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use potrace + FontForge to build a font from generated PNG glyphs.",
    )
    parser.add_argument("--generated-root", required=True,
                        help="Directory containing generated PNGs, or a parent directory that contains one generated/ folder.")
    parser.add_argument("--template-font", default="train/FZNewKai.ttf")
    parser.add_argument("--output-font", required=True)
    parser.add_argument("--family-name", default="Zi2ZiJIT Generated")
    parser.add_argument("--style-name", default="Regular")
    parser.add_argument("--font-version", default="1.000")
    parser.add_argument("--potrace-bin", default="potrace")
    parser.add_argument("--fontforge-bin", default="fontforge")
    parser.add_argument("--potrace-turdsize", type=int, default=2)
    parser.add_argument("--potrace-alphamax", type=float, default=1.0)
    parser.add_argument("--potrace-opttolerance", type=float, default=0.2)
    parser.add_argument("--keep-ungenerated-cjk", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def resolve_generated_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Generated root not found: {root}")

    direct_pngs = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png") if root.is_dir() else []
    if direct_pngs:
        return root

    candidates = []
    for candidate in root.rglob("generated"):
        if candidate.is_dir() and any(p.suffix.lower() == ".png" for p in candidate.iterdir()):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No generated PNG directory found under {root}. "
            "Pass either the generated/ folder itself or a parent directory containing exactly one generated/ folder."
        )
    if len(candidates) > 1:
        names = "\n".join(str(path) for path in candidates[:10])
        raise RuntimeError(
            f"Multiple generated folders found under {root}. Please pass one explicitly:\n{names}"
        )
    return candidates[0]


def parse_codepoint_from_name(path: Path) -> int | None:
    stem = path.stem.upper()
    match = re.search(r"U\+([0-9A-F]{4,6})", stem)
    if match:
        try:
            return int(match.group(1), 16)
        except ValueError:
            return None
    if len(stem) == 1:
        return ord(stem)
    return None


def detect_foreground_mask(gray: np.ndarray) -> np.ndarray:
    _, inv_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground_ratio = float(inv_mask.mean()) / 255.0
    if foreground_ratio <= 0.5:
        return inv_mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def sanitize_postscript_name(family_name: str, style_name: str, output_font: Path) -> str:
    base = f"{family_name}-{style_name}".strip("-")
    ascii_only = re.sub(r"[^A-Za-z0-9-]+", "-", base).strip("-")
    if not ascii_only:
        ascii_only = re.sub(r"[^A-Za-z0-9-]+", "-", output_font.stem).strip("-")
    if not ascii_only:
        ascii_only = "Zi2ZiJIT-Generated"
    return ascii_only[:63]


def ensure_external_tools(args: argparse.Namespace) -> tuple[str, str]:
    potrace = shutil.which(args.potrace_bin)
    fontforge = shutil.which(args.fontforge_bin)
    if args.dry_run:
        return potrace or args.potrace_bin, fontforge or args.fontforge_bin
    if potrace is None:
        raise FileNotFoundError(
            f"Could not find potrace executable '{args.potrace_bin}'. "
            "On Ubuntu, install it with: sudo apt-get install potrace"
        )
    if fontforge is None:
        raise FileNotFoundError(
            f"Could not find FontForge executable '{args.fontforge_bin}'. "
            "On Ubuntu, install it with: sudo apt-get install fontforge"
        )
    return potrace, fontforge


def create_potrace_bitmap(image_path: Path, pbm_path: Path) -> None:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read generated glyph image: {image_path}")
    mask = detect_foreground_mask(gray)
    bitmap = 255 - mask
    pbm_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bitmap).convert("1").save(pbm_path)


def run_potrace(
    potrace_bin: str,
    pbm_path: Path,
    svg_path: Path,
    *,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        potrace_bin,
        str(pbm_path),
        "-s",
        "-o",
        str(svg_path),
        "--turdsize",
        str(turdsize),
        "--alphamax",
        str(alphamax),
        "--opttolerance",
        str(opttolerance),
    ]
    subprocess.run(cmd, check=True)


def build_manifest(
    *,
    generated_dir: Path,
    template_font_path: Path,
    output_font_path: Path,
    family_name: str,
    style_name: str,
    font_version: str,
    svg_dir: Path,
    keep_ungenerated_cjk: bool,
    strict: bool,
) -> dict:
    template_font = TTFont(str(template_font_path))
    cmap = template_font.getBestCmap() or {}
    hmtx_metrics = template_font["hmtx"].metrics

    generated_glyphs = []
    generated_codepoints: set[int] = set()
    missing_cmap = []
    skipped = []

    image_paths = sorted(p for p in generated_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in generated directory: {generated_dir}")

    for image_path in image_paths:
        codepoint = parse_codepoint_from_name(image_path)
        if codepoint is None:
            skipped.append({"file": image_path.name, "reason": "unsupported filename"})
            continue

        glyph_name = cmap.get(codepoint)
        if glyph_name is None:
            missing_cmap.append({"file": image_path.name, "codepoint": f"U+{codepoint:04X}"})
            if strict:
                raise KeyError(f"Codepoint U+{codepoint:04X} is not present in template font {template_font_path}.")
            continue

        generated_codepoints.add(codepoint)
        generated_glyphs.append(
            {
                "codepoint": codepoint,
                "glyph_name": glyph_name,
                "image_path": str(image_path),
                "svg_path": str(svg_dir / f"U+{codepoint:04X}.svg"),
                "width": int(hmtx_metrics[glyph_name][0]),
            }
        )

    clear_cjk_codepoints = []
    if not keep_ungenerated_cjk:
        for codepoint, glyph_name in sorted(cmap.items()):
            if not is_cjk_codepoint(codepoint):
                continue
            if codepoint in generated_codepoints:
                continue
            clear_cjk_codepoints.append(
                {
                    "codepoint": codepoint,
                    "glyph_name": glyph_name,
                    "width": int(hmtx_metrics[glyph_name][0]),
                }
            )

    full_name = f"{family_name} {style_name}".strip()
    manifest = {
        "template_font": str(template_font_path),
        "output_font": str(output_font_path),
        "family_name": family_name,
        "style_name": style_name,
        "full_name": full_name,
        "postscript_name": sanitize_postscript_name(family_name, style_name, output_font_path),
        "font_version": font_version,
        "generated_glyphs": generated_glyphs,
        "clear_cjk_codepoints": clear_cjk_codepoints,
        "missing_cmap": missing_cmap,
        "skipped": skipped,
    }
    return manifest


def main() -> None:
    args = parse_args()
    generated_root = resolve_path(args.generated_root)
    generated_dir = resolve_generated_dir(generated_root)
    template_font_path = resolve_path(args.template_font)
    output_font_path = resolve_path(args.output_font)
    report_json_path = resolve_path(args.report_json) if args.report_json else None
    potrace_bin, fontforge_bin = ensure_external_tools(args)

    if not template_font_path.is_file():
        raise FileNotFoundError(f"Template font not found: {template_font_path}")

    work_root = output_font_path.parent / f".{output_font_path.stem}_fontforge_tmp"
    bitmap_dir = work_root / "pbm"
    svg_dir = work_root / "svg"
    manifest_path = work_root / "manifest.json"

    manifest = build_manifest(
        generated_dir=generated_dir,
        template_font_path=template_font_path,
        output_font_path=output_font_path,
        family_name=args.family_name,
        style_name=args.style_name,
        font_version=args.font_version,
        svg_dir=svg_dir,
        keep_ungenerated_cjk=args.keep_ungenerated_cjk,
        strict=args.strict,
    )

    print("FontForge build plan:")
    print(f"  generated_dir      = {generated_dir}")
    print(f"  template_font      = {template_font_path}")
    print(f"  output_font        = {output_font_path}")
    print(f"  generated_glyphs   = {len(manifest['generated_glyphs'])}")
    print(f"  cleared_cjk_glyphs = {len(manifest['clear_cjk_codepoints'])}")
    if manifest["missing_cmap"]:
        print(f"  missing_in_template = {len(manifest['missing_cmap'])}")
    if manifest["skipped"]:
        print(f"  skipped_files      = {len(manifest['skipped'])}")

    if args.dry_run:
        return

    work_root.mkdir(parents=True, exist_ok=True)
    for item in manifest["generated_glyphs"]:
        image_path = Path(item["image_path"])
        svg_path = Path(item["svg_path"])
        pbm_path = bitmap_dir / f"U+{item['codepoint']:04X}.pbm"
        create_potrace_bitmap(image_path, pbm_path)
        run_potrace(
            potrace_bin,
            pbm_path,
            svg_path,
            turdsize=args.potrace_turdsize,
            alphamax=args.potrace_alphamax,
            opttolerance=args.potrace_opttolerance,
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    output_font_path.parent.mkdir(parents=True, exist_ok=True)
    fontforge_script = REPO_ROOT / "scripts" / "_fontforge_build_from_manifest.py"
    cmd = [
        fontforge_bin,
        "-lang=py",
        "-script",
        str(fontforge_script),
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    report = {
        "generated_root": str(generated_root),
        "generated_dir": str(generated_dir),
        "template_font": str(template_font_path),
        "output_font": str(output_font_path),
        "family_name": args.family_name,
        "style_name": args.style_name,
        "font_version": args.font_version,
        "generated_glyphs_count": len(manifest["generated_glyphs"]),
        "generated_glyphs": manifest["generated_glyphs"],
        "cleared_cjk_glyphs_count": len(manifest["clear_cjk_codepoints"]),
        "cleared_cjk_glyphs": manifest["clear_cjk_codepoints"],
        "missing_cmap_count": len(manifest["missing_cmap"]),
        "missing_cmap": manifest["missing_cmap"],
        "skipped_count": len(manifest["skipped"]),
        "skipped": manifest["skipped"],
        "work_root": str(work_root),
    }
    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.keep_temp:
        shutil.rmtree(work_root, ignore_errors=True)

    print("FontForge build complete.")
    print(f"  output_font        = {output_font_path}")
    print(f"  generated_glyphs   = {len(manifest['generated_glyphs'])}")
    print(f"  cleared_cjk_glyphs = {len(manifest['clear_cjk_codepoints'])}")
    if report_json_path is not None:
        print(f"  report_json        = {report_json_path}")


if __name__ == "__main__":
    main()
