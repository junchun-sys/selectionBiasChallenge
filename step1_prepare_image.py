"""
Step 1: Prepare a black and white image for the statistics meme.
Loads an image, converts to grayscale, and resizes to appropriate dimensions
while maintaining aspect ratio.
"""

import numpy as np
from PIL import Image


def prepare_image(
    img_path: str,
    max_size: int = 512,
    target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Load an image, convert to grayscale, and resize to appropriate dimensions
    for the statistics meme while maintaining aspect ratio.
    
    Parameters
    ----------
    img_path : str
        Path to the input image file
    max_size : int
        Maximum dimension (width or height) if target_size is None.
        Image will be resized to fit within this size while maintaining aspect ratio.
    target_size : tuple[int, int] | None
        Optional target size (width, height). If provided, image will be resized
        to this size. If None, uses max_size to determine dimensions.
    
    Returns
    -------
    img_array : np.ndarray
        Grayscale image as 2D array (height, width) with values in [0, 1]
    """
    # Load the image
    original_img = Image.open(img_path)
    
    # Convert to grayscale if needed
    if original_img.mode != 'L':
        original_img = original_img.convert('L')
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(original_img, dtype=np.float32) / 255.0
    
    # Resize if needed
    if target_size is not None:
        # Resize to exact target size
        new_size = target_size
        img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
        if img_resized_pil.mode != 'L':
            img_resized_pil = img_resized_pil.convert('L')
        img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
    elif img_array.shape[0] > max_size or img_array.shape[1] > max_size:
        # Resize to fit within max_size while maintaining aspect ratio
        scale = max_size / max(img_array.shape[0], img_array.shape[1])
        new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
        img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
        if img_resized_pil.mode != 'L':
            img_resized_pil = img_resized_pil.convert('L')
        img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
    else:
        img_resized = img_array.copy()
    
    # Ensure img_resized is 2D grayscale
    if len(img_resized.shape) > 2:
        img_resized = img_resized[:, :, 0]
    elif len(img_resized.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img_resized.shape}")
    
    return img_resized

