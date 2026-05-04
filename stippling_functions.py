"""
Blue noise stippling functions using a modified void-and-cluster algorithm.
"""

import numpy as np
from importance_map import compute_importance


def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
    """
    Create a periodic (toroidal) 2D Gaussian kernel centered at (0,0).
    The toroidal property means the kernel wraps around at the edges,
    ensuring consistent repulsion behavior regardless of point location.
    
    Parameters:
    -----------
    h : int
        Height of the kernel (should match image height)
    w : int
        Width of the kernel (should match image width)
    sigma : float
        Standard deviation of the Gaussian (controls repulsion radius)
    
    Returns:
    --------
    kern : np.ndarray
        Normalized 2D Gaussian kernel with toroidal wrapping
    """
    y = np.arange(h)
    x = np.arange(w)
    # Compute toroidal distances (minimum distance considering wrapping)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    # Compute Gaussian
    kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kern.sum()
    if s > 0:
        kern /= s  # Normalize
    return kern


def _toroidal_splat_grids(
    h: int, w: int, kernel: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build offset grids and per-offset Gaussian weights for truncated toroidal splats.

    Weight at offset (dy, dx) equals ``kernel[(dy + h) % h, (dx + w) % w]`` and does
    not depend on splat center, so it is computed once.
    """
    r = int(min(radius, h - 1, w - 1))
    dy_grid, dx_grid = np.meshgrid(
        np.arange(-r, r + 1, dtype=np.int32),
        np.arange(-r, r + 1, dtype=np.int32),
        indexing="ij",
    )
    ki = (dy_grid.astype(np.int64, copy=False) + h) % h
    kj = (dx_grid.astype(np.int64, copy=False) + w) % w
    splat_w = kernel[ki, kj].astype(np.float64, copy=False)
    return dy_grid.astype(np.int64, copy=False), dx_grid.astype(np.int64, copy=False), splat_w


def _toroidal_splat_add_inplace(
    energy: np.ndarray,
    y0: int,
    x0: int,
    h: int,
    w: int,
    dy_grid: np.ndarray,
    dx_grid: np.ndarray,
    splat_w: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
) -> None:
    """Add a truncated toroidal Gaussian splat centered at (y0, x0) into energy."""
    np.add(y0, dy_grid, out=yy, casting="unsafe")
    np.mod(yy, h, out=yy)
    np.add(x0, dx_grid, out=xx, casting="unsafe")
    np.mod(xx, w, out=xx)
    energy[yy, xx] += splat_w


def void_and_cluster(
    input_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    importance_img: np.ndarray | None = None,
    noise_scale_factor: float = 0.1,
):
    """
    Generate blue noise stippling pattern from input image using a modified
    void-and-cluster algorithm with content-weighted importance.
    
    Parameters:
    -----------
    input_img : np.ndarray
        Input image as 2D array (grayscale, normalized to [0, 1])
    percentage : float
        Percentage of pixels to stipple (0.0 to 1.0). Lower values (0.05-0.12)
        create sparser, more focused patterns.
    sigma : float
        Standard deviation of Gaussian kernel for repulsion (in pixels).
        Controls the minimum spacing between stipples.
    content_bias : float
        Scales the importance of image content in the energy field.
        Higher values (0.8-0.95) prioritize following the importance map;
        lower values allow more uniform spatial distribution.
    importance_img : np.ndarray | None
        Optional precomputed importance map (same shape as input).
        If None, importance is computed automatically from the input image.
    noise_scale_factor : float
        Scale factor for exploration noise (lower = crisper features, less exploration).
        Values typically range from 0.05 to 0.2.
    
    Returns:
    --------
    final_stipple : np.ndarray
        Binary stippling pattern (0.0 = black dot, 1.0 = white background)
    samples : np.ndarray
        Array of (y, x, intensity) tuples for each stipple point
    """
    I = np.clip(input_img, 0.0, 1.0)
    h, w = I.shape

    # Compute or use provided importance map
    if importance_img is None:
        importance = compute_importance(I)
    else:
        importance = np.clip(importance_img, 0.0, 1.0)

    # Create toroidal Gaussian kernel for repulsion
    kernel = toroidal_gaussian_kernel(h, w, sigma)

    # Truncation radius: splat energy outside this ring is negligible vs float32 noise
    splat_radius = int(
        min(
            h - 1,
            w - 1,
            max(4, int(np.ceil(6.0 * float(sigma)))),
        )
    )
    dy_grid, dx_grid, splat_w = _toroidal_splat_grids(h, w, kernel, splat_radius)
    yy_buf = np.empty_like(dy_grid, dtype=np.int64)
    xx_buf = np.empty_like(dx_grid, dtype=np.int64)

    # Initialize energy field: lower energy → more likely to be picked
    energy_current = (-importance * content_bias).astype(np.float64, copy=False)

    # Stipple buffer: start with white background; selected points become black dots
    final_stipple = np.ones_like(I)
    samples = []

    # Number of points to select
    num_points = int(I.size * percentage)

    # Preallocate scratch buffers (avoids allocating full H×W arrays each iteration)
    noise = np.empty((h, w), dtype=np.float64)
    energy_for_argmin = np.empty((h, w), dtype=np.float64)
    rng = np.random.default_rng()

    # Choose first point near center with minimal energy
    cy, cx = h // 2, w // 2
    r = min(20, h // 10, w // 10)
    ys = slice(max(0, cy - r), min(h, cy + r))
    xs = slice(max(0, cx - r), min(w, cx + r))
    region = energy_current[ys, xs]
    flat = np.argmin(region)
    y0 = flat // (region.shape[1]) + (cy - r)
    x0 = flat % (region.shape[1]) + (cx - r)

    # Place first point
    _toroidal_splat_add_inplace(
        energy_current, y0, x0, h, w, dy_grid, dx_grid, splat_w, yy_buf, xx_buf
    )
    energy_current[y0, x0] = np.inf  # Prevent reselection
    samples.append((y0, x0, I[y0, x0]))
    final_stipple[y0, x0] = 0.0  # Black dot

    scale_base = noise_scale_factor * content_bias

    # Iteratively place remaining points
    for i in range(1, num_points):
        # Add exploration noise that decreases over time
        exploration = 1.0 - (i / num_points) * 0.5  # Decrease from 1.0 to 0.5
        rng.standard_normal(out=noise)
        np.multiply(noise, scale_base * exploration, out=noise)
        np.add(energy_current, noise, out=energy_for_argmin)

        # Find position with minimum energy (with noise for exploration)
        pos_flat = int(np.argmin(energy_for_argmin))
        y = pos_flat // w
        x = pos_flat % w

        # Add Gaussian splat to prevent nearby points from being selected
        _toroidal_splat_add_inplace(
            energy_current, y, x, h, w, dy_grid, dx_grid, splat_w, yy_buf, xx_buf
        )
        energy_current[y, x] = np.inf  # Prevent reselection

        # Record the sample
        samples.append((y, x, I[y, x]))
        final_stipple[y, x] = 0.0  # Black dot

    return final_stipple, np.array(samples)

