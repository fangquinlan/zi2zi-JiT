#!/usr/bin/env python3
"""Build a TTF font from generated glyph PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _launcher_utils import maybe_reexec_with_repo_python

maybe_reexec_with_repo_python(__file__, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated U+XXXX.png glyphs into a TTF font using a template font.",
    )
    parser.add_argument("--generated-root", required=True,
                        help="Directory containing generated PNGs, or a parent directory that contains one generated/ folder.")
    parser.add_argument("--template-font", default="train/FZNewKai.ttf",
                        help="Template TTF used for cmap, metrics, and non-generated glyphs.")
    parser.add_argument("--output-font", required=True,
                        help="Output TTF file path.")
    parser.add_argument("--family-name", default="Zi2ZiJIT Generated",
                        help="Font family name to write into the output font.")
    parser.add_argument("--style-name", default="Regular",
                        help="Font style/subfamily name.")
    parser.add_argument("--font-version", default="1.000",
                        help="Font version string.")
    parser.add_argument("--approx-tolerance", type=float, default=1.5,
                        help="Polygon simplification tolerance in pixels.")
    parser.add_argument("--min-area", type=float, default=8.0,
                        help="Minimum contour area in image pixels to keep.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if a generated image does not map to a glyph in the template font.")
    parser.add_argument("--report-json", default="",
                        help="Optional JSON report output path.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def resolve_generated_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Generated root not found: {root}")

    pngs = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png") if root.is_dir() else []
    if pngs:
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
    if stem.startswith("U+"):
        try:
            return int(stem[2:], 16)
        except ValueError:
            return None
    if len(stem) == 1:
        return ord(stem)
    return None


def detect_foreground_mask(gray):
    _, inv_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground_ratio = float(inv_mask.mean()) / 255.0
    if foreground_ratio <= 0.5:
        return inv_mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def contour_depth(index: int, hierarchy) -> int:
    depth = 0
    parent = hierarchy[index][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth


def polygon_signed_area(points: list[tuple[int, int]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def simplify_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    simplified: list[tuple[int, int]] = []
    for point in points:
        if not simplified or simplified[-1] != point:
            simplified.append(point)
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified.pop()
    return simplified


def image_point_to_font(point, width: int, height: int, units_per_em: int) -> tuple[int, int]:
    x = int(round((float(point[0]) / max(width - 1, 1)) * units_per_em))
    y = int(round(((height - 1 - float(point[1])) / max(height - 1, 1)) * units_per_em))
    return x, y


def glyph_from_png(image_path: Path, units_per_em: int, approx_tolerance: float, min_area: float):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read glyph image: {image_path}")

    mask = detect_foreground_mask(gray)
    contours_data = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_data) == 3:
        _, contours, hierarchy = contours_data
    else:
        contours, hierarchy = contours_data

    pen = TTGlyphPen(None)
    if hierarchy is None or len(contours) == 0:
        return pen.glyph()

    hierarchy = hierarchy[0]
    kept = 0
    height, width = gray.shape[:2]

    for idx, contour in enumerate(contours):
        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        epsilon = approx_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        points = [image_point_to_font(point, width, height, units_per_em) for point in approx]
        points = simplify_points(points)
        if len(points) < 3:
            continue

        depth = contour_depth(idx, hierarchy)
        want_clockwise = depth % 2 == 0
        signed_area = polygon_signed_area(points)
        is_clockwise = signed_area < 0
        if want_clockwise != is_clockwise:
            points = list(reversed(points))

        pen.moveTo(points[0])
        for point in points[1:]:
            pen.lineTo(point)
        pen.closePath()
        kept += 1

    if kept == 0:
        return TTGlyphPen(None).glyph()
    return pen.glyph()


def set_name_record(name_table, name_id: int, value: str) -> None:
    for platform_id, plat_enc_id, lang_id in ((3, 1, 0x409), (1, 0, 0)):
        name_table.setName(value, name_id, platform_id, plat_enc_id, lang_id)


def update_font_names(font: TTFont, family_name: str, style_name: str, font_version: str) -> None:
    full_name = f"{family_name} {style_name}".strip()
    postscript_name = full_name.replace(" ", "-")
    unique_name = f"{font_version};Zi2ZiJIT;{postscript_name}"
    version_string = f"Version {font_version}"

    name_table = font["name"]
    set_name_record(name_table, 1, family_name)
    set_name_record(name_table, 2, style_name)
    set_name_record(name_table, 3, unique_name)
    set_name_record(name_table, 4, full_name)
    set_name_record(name_table, 5, version_string)
    set_name_record(name_table, 6, postscript_name)


def main() -> None:
    args = parse_args()
    generated_root = resolve_path(args.generated_root)
    generated_dir = resolve_generated_dir(generated_root)
    template_font_path = resolve_path(args.template_font)
    output_font_path = resolve_path(args.output_font)
    report_json_path = resolve_path(args.report_json) if args.report_json else None

    if not template_font_path.is_file():
        raise FileNotFoundError(f"Template font not found: {template_font_path}")

    font = TTFont(str(template_font_path))
    if "glyf" not in font:
        raise RuntimeError(
            f"Template font {template_font_path} does not contain a glyf table. "
            "Please use a TrueType (.ttf) font as the template."
        )

    cmap = font.getBestCmap() or {}
    units_per_em = int(font["head"].unitsPerEm)
    glyf_table = font["glyf"]
    hmtx_table = font["hmtx"]

    replaced = []
    skipped = []
    missing_cmap = []

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
            if args.strict:
                raise KeyError(f"Codepoint U+{codepoint:04X} is not present in template font {template_font_path}.")
            continue

        glyph = glyph_from_png(
            image_path,
            units_per_em=units_per_em,
            approx_tolerance=args.approx_tolerance,
            min_area=args.min_area,
        )
        glyf_table[glyph_name] = glyph
        glyph.recalcBounds(glyf_table)
        advance_width, _ = hmtx_table.metrics[glyph_name]
        left_side_bearing = getattr(glyph, "xMin", 0) if getattr(glyph, "numberOfContours", 0) != 0 else 0
        hmtx_table.metrics[glyph_name] = (advance_width, left_side_bearing)
        replaced.append({
            "file": image_path.name,
            "codepoint": f"U+{codepoint:04X}",
            "glyph_name": glyph_name,
        })

    update_font_names(
        font,
        family_name=args.family_name,
        style_name=args.style_name,
        font_version=args.font_version,
    )

    font.recalcBBoxes = True
    font.recalcTimestamp = True
    if hasattr(font["maxp"], "recalc"):
        font["maxp"].recalc(font)

    output_font_path.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(output_font_path))

    report = {
        "generated_root": str(generated_root),
        "generated_dir": str(generated_dir),
        "template_font": str(template_font_path),
        "output_font": str(output_font_path),
        "family_name": args.family_name,
        "style_name": args.style_name,
        "font_version": args.font_version,
        "replaced_count": len(replaced),
        "replaced": replaced,
        "missing_cmap_count": len(missing_cmap),
        "missing_cmap": missing_cmap,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("Font build complete.")
    print(f"  generated_dir  = {generated_dir}")
    print(f"  template_font  = {template_font_path}")
    print(f"  output_font    = {output_font_path}")
    print(f"  replaced_glyphs = {len(replaced)}")
    if missing_cmap:
        print(f"  missing_in_template = {len(missing_cmap)}")
    if skipped:
        print(f"  skipped_files = {len(skipped)}")
    if report_json_path is not None:
        print(f"  report_json    = {report_json_path}")


if __name__ == "__main__":
    main()
