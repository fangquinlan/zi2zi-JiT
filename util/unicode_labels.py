from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def parse_unicode_codepoint_from_name(name: str) -> int | None:
    stem = Path(name).stem.upper()
    match = re.search(r"U\+([0-9A-F]{4,6})", stem)
    if not match:
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


def normalize_unicode_codepoints(values: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, int):
            normalized.add(value)
            continue
        text = str(value).strip().upper()
        if not text:
            continue
        if text.startswith("U+"):
            text = text[2:]
        normalized.add(int(text, 16))
    return sorted(normalized)


def build_unicode_index_map(codepoints: Iterable[int | str]) -> dict[int, int]:
    normalized = normalize_unicode_codepoints(codepoints)
    return {codepoint: idx for idx, codepoint in enumerate(normalized)}


def load_unicode_codepoints(path_text: str) -> list[int]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Unicode codepoint metadata not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "unicode_codepoints" in payload:
                return normalize_unicode_codepoints(payload["unicode_codepoints"])
            raise KeyError(
                f"JSON file {path} does not contain a 'unicode_codepoints' field."
            )
        if isinstance(payload, list):
            return normalize_unicode_codepoints(payload)
        raise ValueError(f"Unsupported JSON payload in {path}: expected dict or list.")

    with open(path, "r", encoding="utf-8") as f:
        values = [line.strip() for line in f if line.strip()]
    return normalize_unicode_codepoints(values)
