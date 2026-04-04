import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Set, Tuple

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_FONT_SUFFIXES = {".ttf", ".otf", ".woff", ".woff2", ".TTF", ".OTF", ".WOFF", ".WOFF2"}
WEB_FONT_SUFFIXES = {".woff", ".woff2", ".WOFF", ".WOFF2"}

CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x30000, 0x3134F),  # Extension G
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # Compatibility Supplement
]

KANA_RANGES = [
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),    # Halfwidth Katakana
]


def ensure_output_directory(output_path: str) -> Path:
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_font_file(font_path: str) -> Path:
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {font_path}")
    if path.suffix not in SUPPORTED_FONT_SUFFIXES:
        raise ValueError(
            f"Unsupported font format: {path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FONT_SUFFIXES))}"
        )
    return path


def load_font(font_path: str) -> Tuple[TTFont, Path]:
    validated_path = validate_font_file(font_path)
    font = TTFont(str(validated_path))
    return font, validated_path


def _font_cache_dir() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "zi2zi_jit_font_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _font_cache_key(font_path: Path) -> str:
    stat = font_path.stat()
    payload = f"{font_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def prepare_font_for_pil(font_path: str) -> Path:
    validated_path = validate_font_file(font_path)
    if validated_path.suffix not in WEB_FONT_SUFFIXES:
        return validated_path

    cache_dir = _font_cache_dir()
    cache_key = _font_cache_key(validated_path)

    font = TTFont(str(validated_path))
    out_suffix = ".otf" if "CFF " in font else ".ttf"
    cached_path = cache_dir / f"{validated_path.stem}-{cache_key}{out_suffix}"
    if cached_path.exists():
        return cached_path

    font.flavor = None
    font.save(str(cached_path))
    return cached_path


def is_cjk_codepoint(codepoint: int) -> bool:
    for start, end in CJK_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def is_kana_codepoint(codepoint: int) -> bool:
    for start, end in KANA_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def has_valid_outline(font: TTFont, codepoint: int) -> bool:
    cmap = font.getBestCmap()
    if cmap is None or codepoint not in cmap:
        return False

    glyph_name = cmap[codepoint]

    if "glyf" in font:
        glyf = font["glyf"]
        if glyph_name not in glyf:
            return False
        glyph = glyf[glyph_name]
        if glyph.numberOfContours == 0:
            return False
        return True

    if "CFF " in font:
        try:
            from fontTools.pens.boundsPen import BoundsPen

            glyphset = font.getGlyphSet()
            if glyph_name not in glyphset:
                return False
            pen = BoundsPen(glyphset)
            glyphset[glyph_name].draw(pen)
            return pen.bounds is not None
        except Exception:
            return False

    return False


def get_cjk_codepoints(font: TTFont, filter_empty: bool = True) -> Set[int]:
    cmap = font.getBestCmap() or {}
    cjk_codepoints = {cp for cp in cmap.keys() if is_cjk_codepoint(cp)}
    if filter_empty:
        cjk_codepoints = {cp for cp in cjk_codepoints if has_valid_outline(font, cp)}
    return cjk_codepoints


def get_renderable_codepoints(font: TTFont, filter_empty: bool = True) -> Set[int]:
    cmap = font.getBestCmap() or {}
    codepoints = set(cmap.keys())
    if filter_empty:
        codepoints = {cp for cp in codepoints if has_valid_outline(font, cp)}
    return codepoints


def extract_font_name(font: TTFont, fallback_path: Path) -> str:
    name_table = font.get("name")
    if not name_table:
        return fallback_path.stem

    for name_id in (4, 1):
        for record in name_table.names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return fallback_path.stem


class GlyphRenderer:
    SAMPLE_CHARS = [
        "中", "国", "人", "大", "小",
        "一", "二", "三", "四", "五",
        "日", "月", "水", "火", "木",
        "口", "田", "目", "耳", "手",
        "上", "下", "左", "右", "前",
        "東", "西", "南", "北", "道",
        "的", "是", "不", "了", "在",
        "有", "和", "这", "为", "我",
        "龍", "龜", "鬱", "靈", "響",
        "永", "風", "雲", "雨", "雪",
    ]

    def __init__(
        self,
        font_path: str,
        resolution: int,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        sample_size: int = 50,
    ):
        self.font_path = font_path
        self.resolution = resolution
        self.background_color = background_color
        self.text_color = text_color
        self.sample_size = sample_size

        self.font_size = int(resolution * 0.8)
        self.render_font_path = prepare_font_for_pil(font_path)
        self.pil_font = ImageFont.truetype(str(self.render_font_path), self.font_size)

        self._tt_font = TTFont(str(font_path))
        self._cmap = self._tt_font.getBestCmap() or {}
        self._global_offset = self._calculate_global_offset()

    def _get_sample_codepoints(self) -> list[int]:
        available = []
        seen = set()

        def extend_from_codepoints(codepoints) -> bool:
            for cp in codepoints:
                if cp in seen or cp not in self._cmap or not has_valid_outline(self._tt_font, cp):
                    continue
                available.append(cp)
                seen.add(cp)
                if len(available) >= self.sample_size:
                    return True
            return False

        for char in self.SAMPLE_CHARS:
            cp = ord(char)
            if extend_from_codepoints([cp]):
                return available[:self.sample_size]

        if len(available) < self.sample_size:
            kana_in_font = sorted(cp for cp in self._cmap.keys() if is_kana_codepoint(cp))
            if extend_from_codepoints(kana_in_font):
                return available[:self.sample_size]

        if len(available) < self.sample_size:
            cjk_in_font = sorted(cp for cp in self._cmap.keys() if is_cjk_codepoint(cp))
            if extend_from_codepoints(cjk_in_font):
                return available[:self.sample_size]

        if len(available) < self.sample_size:
            any_valid = sorted(self._cmap.keys())
            extend_from_codepoints(any_valid)

        return available[:self.sample_size]

    def _calculate_global_offset(self) -> Tuple[int, int]:
        temp_image = Image.new("RGB", (self.resolution, self.resolution), self.background_color)
        draw = ImageDraw.Draw(temp_image)
        samples = self._get_sample_codepoints()

        if not samples:
            return (self.resolution // 2, self.resolution // 2)

        x_offsets = []
        y_offsets = []
        for cp in samples:
            char = chr(cp)
            try:
                bbox = draw.textbbox((0, 0), char, font=self.pil_font)
            except Exception:
                continue
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w <= 0 or text_h <= 0:
                continue
            x_offsets.append((self.resolution - text_w) // 2 - bbox[0])
            y_offsets.append((self.resolution - text_h) // 2 - bbox[1])

        if not x_offsets:
            return (self.resolution // 2, self.resolution // 2)

        return (int(sum(x_offsets) / len(x_offsets)), int(sum(y_offsets) / len(y_offsets)))

    def render(self, codepoint: int) -> Optional[Image.Image]:
        try:
            image = Image.new("RGB", (self.resolution, self.resolution), self.background_color)
            draw = ImageDraw.Draw(image)
            char = chr(codepoint)
            try:
                bbox = draw.textbbox((0, 0), char, font=self.pil_font)
            except Exception:
                bbox = None
            if bbox is not None:
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w > 0 and text_h > 0:
                    offset = (
                        (self.resolution - text_w) // 2 - bbox[0],
                        (self.resolution - text_h) // 2 - bbox[1],
                    )
                else:
                    offset = self._global_offset
            else:
                offset = self._global_offset
            draw.text(offset, char, font=self.pil_font, fill=self.text_color)
            return image
        except Exception:
            return None
