#!/usr/bin/env python3
"""Run under FontForge: import SVG outlines into a template font and generate TTF."""

import json
import sys

import fontforge


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: fontforge -lang=py -script _fontforge_build_from_manifest.py <manifest.json>")

    manifest = load_manifest(sys.argv[1])
    font = fontforge.open(manifest["template_font"])
    font.encoding = "UnicodeFull"

    for item in manifest.get("clear_cjk_codepoints", []):
        codepoint = int(item["codepoint"])
        glyph = font.createChar(codepoint)
        glyph.clear()
        glyph.width = int(item["width"])

    for item in manifest.get("generated_glyphs", []):
        codepoint = int(item["codepoint"])
        glyph = font.createChar(codepoint)
        glyph.clear()
        glyph.importOutlines(item["svg_path"])
        glyph.removeOverlap()
        glyph.correctDirection()
        glyph.simplify()
        glyph.round()
        glyph.width = int(item["width"])

    font.familyname = manifest["family_name"]
    font.fullname = manifest["full_name"]
    font.fontname = manifest["postscript_name"]
    font.version = manifest["font_version"]

    font.generate(manifest["output_font"])
    font.close()


if __name__ == "__main__":
    main()
