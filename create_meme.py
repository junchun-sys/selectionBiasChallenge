"""Assemble the four-panel statistics meme image."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image

# Soft cherry-blossom pink background (hex matches common "sakura" tints)
SAKURA_PINK: str = "#FFE8F0"

__all__ = ["SAKURA_PINK", "create_statistics_meme"]

# Pillow 9.1+ uses Image.Resampling; older versions use Image.LANCZOS
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def _normalize_gray(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Each image must be a 2D array (height × width)")
    return np.clip(a, 0.0, 1.0)


def _resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize grayscale float [0,1] to (height, width) = target_shape."""
    if arr.shape == target_shape:
        return arr
    h, w = target_shape
    img = Image.fromarray((_normalize_gray(arr) * 255.0).round().astype(np.uint8), mode="L")
    img = img.resize((w, h), _RESAMPLE)
    return np.clip(np.asarray(img, dtype=np.float64) / 255.0, 0.0, 1.0)


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = SAKURA_PINK,
) -> None:
    """
    Save a 1×4 panel PNG (no panel captions on the figure).

    All panels are resized to match ``original_img`` shape if needed.
    Default background is soft cherry-blossom pink (``SAKURA_PINK``); pass
    ``"white"`` or any matplotlib color string / hex for other looks.
    """
    target_shape = original_img.shape
    panels = [
        _resize_to_shape(_normalize_gray(original_img), target_shape),
        _resize_to_shape(_normalize_gray(stipple_img), target_shape),
        _resize_to_shape(_normalize_gray(block_letter_img), target_shape),
        _resize_to_shape(_normalize_gray(masked_stipple_img), target_shape),
    ]
    try:
        face = mcolors.to_rgb(background_color)
    except ValueError:
        face = mcolors.to_rgb(SAKURA_PINK)

    # Wide strip: four equal panels, labels on top (matches course example figure)
    fig_w, fig_h = 16.0, 4.5
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(fig_w, fig_h),
        facecolor=face,
    )
    fig.patch.set_facecolor(face)

    for ax, img in zip(axes, panels):
        ax.set_facecolor(face)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.55")

    plt.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.08, wspace=0.14)
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=face,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.15,
        format="png",
    )
    plt.close(fig)
