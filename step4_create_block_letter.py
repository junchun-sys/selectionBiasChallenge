"""
Step 4: Render a block letter (default "S") as a grayscale mask matching image dimensions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _candidate_font_paths() -> list[Path]:
    """Ordered list of likely bold system font paths (Windows, macOS, Linux)."""
    paths: list[Path] = []
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        fonts = windir / "Fonts"
        paths.extend(
            [
                fonts / "arialbd.ttf",
                fonts / "calibrib.ttf",
                fonts / "segoeuib.ttf",
                fonts / "verdanab.ttf",
            ]
        )
    elif sys.platform == "darwin":
        paths.extend(
            [
                Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                Path("/Library/Fonts/Arial Bold.ttf"),
                Path("/System/Library/Fonts/Helvetica.ttc"),
            ]
        )
    else:
        paths.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            ]
        )
    return paths


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a bold TrueType font at ``size``, or fall back to default bitmap font."""
    for path in _candidate_font_paths():
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_pixel_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    # Bitmap default font
    return font.getsize(text)


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single letter centered on white, return grayscale ``[0, 1]`` (height × width).

    Parameters
    ----------
    height, width : int
        Output array shape (rows, cols).
    letter : str
        Character(s) to draw; default ``"S"``.
    font_size_ratio : float
        Target fraction of ``min(height, width)`` used to bound glyph size (largest
        font that still fits within that box).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not letter:
        raise ValueError("letter must be non-empty")
    ratio = max(0.05, min(1.0, float(font_size_ratio)))
    max_w = int(width * ratio)
    max_h = int(height * ratio)
    max_w = max(max_w, 1)
    max_h = max(max_h, 1)

    # Largest font size that fits inside the box (coarse search is enough)
    lo = 8
    hi = max(int(min(height, width) * ratio) + 2, lo)
    best_font = _load_bold_font(lo)
    for size in range(hi, lo - 1, -1):
        font = _load_bold_font(size)
        tw, th = _text_pixel_size(font, letter)
        if tw <= max_w and th <= max_h and tw > 0 and th > 0:
            best_font = font
            break

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    cx, cy = width / 2.0, height / 2.0
    try:
        draw.text((cx, cy), letter, fill=0, font=best_font, anchor="mm")
    except TypeError:
        # Older Pillow without anchor: manual centering
        tw, th = _text_pixel_size(best_font, letter)
        if hasattr(best_font, "getbbox"):
            left, top, _, _ = best_font.getbbox(letter)
        else:
            left, top = 0, 0
        x = int(round(cx - tw / 2 - left))
        y = int(round(cy - th / 2 - top))
        draw.text((x, y), letter, fill=0, font=best_font)

    arr = np.asarray(img, dtype=np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0)
