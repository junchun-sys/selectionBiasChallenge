"""
Step 5: Apply the block-letter mask to the stippled image.
Pixels in the masked (dark) region are set to white, simulating removed data points.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply a mask to a stippled image: dark mask regions become white (stipples removed).

    Parameters
    ----------
    stipple_img : np.ndarray
        2D stippled image (height × width), typically values in [0, 1].
    mask_img : np.ndarray
        2D mask with values in [0, 1]: 0.0 = black (masked / letter area), 1.0 = white (keep).
    threshold : float
        Pixels with mask value strictly below this are treated as masked (set to 1.0).

    Returns
    -------
    np.ndarray
        2D array with the same shape as the inputs.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"stipple_img shape {stipple_img.shape} must match mask_img shape {mask_img.shape}"
        )
    white = np.ones_like(stipple_img, dtype=stipple_img.dtype)
    return np.where(mask_img < threshold, white, stipple_img)
