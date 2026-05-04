"""
Step 2: Create a stippled image from the prepared grayscale image.
Uses blue noise stippling to convert the image into a pattern of dots.
"""

import numpy as np
from importance_map import compute_importance
from stippling_functions import void_and_cluster


def create_stipple(
    gray_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    noise_scale_factor: float = 0.1,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.2,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a blue noise stippling pattern from a grayscale image.
    
    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image as 2D array (height, width) with values in [0, 1]
    percentage : float
        Percentage of pixels to stipple (0.0 to 1.0). Default 0.08 (8%).
    sigma : float
        Standard deviation of Gaussian kernel for repulsion (in pixels).
        Controls the minimum spacing between stipples. Default 0.9.
    content_bias : float
        Scales the importance of image content in the energy field.
        Higher values (0.8-0.95) prioritize following the importance map.
        Default 0.9.
    noise_scale_factor : float
        Scale factor for exploration noise. Default 0.1.
    extreme_downweight : float
        Strength of downweighting for extreme tones. Default 0.5.
    extreme_threshold_low : float
        Threshold below which tones are considered "very dark". Default 0.2.
    extreme_threshold_high : float
        Threshold above which tones are considered "very light". Default 0.8.
    extreme_sigma : float
        Width of the smooth transition for extreme downweighting. Default 0.1.
    
    Returns
    -------
    stipple_pattern : np.ndarray
        Binary stippling pattern (0.0 = black dot, 1.0 = white background)
        Same shape as input image
    samples : np.ndarray
        Array of (y, x, intensity) tuples for each stipple point
    """
    # Compute importance map
    importance_map = compute_importance(
        gray_img,
        extreme_downweight=extreme_downweight,
        extreme_threshold_low=extreme_threshold_low,
        extreme_threshold_high=extreme_threshold_high,
        extreme_sigma=extreme_sigma
    )
    
    # Generate stippling pattern
    stipple_pattern, samples = void_and_cluster(
        gray_img,
        percentage=percentage,
        sigma=sigma,
        content_bias=content_bias,
        importance_img=importance_map,
        noise_scale_factor=noise_scale_factor
    )
    
    return stipple_pattern, samples

