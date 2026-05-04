"""
Step 3: Create a tonal analysis version of the image.
This computes box-averaged tones across a grid to help understand the
distribution of brightness values, which can be used to tune stippling parameters.
"""

import numpy as np


def create_tonal(
    gray_img: np.ndarray,
    grid_rows: int = 16,
    grid_cols: int = 12,
    return_full_image: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a tonal analysis by dividing the image into a grid and computing
    average brightness in each section. This helps identify tone distribution
    and can be used to tune stippling parameters in Step 2.
    
    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image as 2D array (height, width) with values in [0, 1]
    grid_rows : int
        Number of rows in the analysis grid. Default 16.
    grid_cols : int
        Number of columns in the analysis grid. Default 12.
    return_full_image : bool
        If True, returns a full-size image with box-averaged values expanded
        to fill each grid cell. If False, returns just the grid array.
        Default True.
    
    Returns
    -------
    tonal_img : np.ndarray
        If return_full_image=True: Full-size image (height, width) where each
        grid cell is filled with the average tone value for that section.
        If return_full_image=False: Grid array (grid_rows, grid_cols) with
        average tone values.
    average_tones : np.ndarray
        Grid array (grid_rows, grid_cols) with average tone values for each section
    stats : dict
        Dictionary containing tonal statistics:
        - 'mean': Overall mean tone
        - 'std': Overall standard deviation
        - 'min': Minimum tone value
        - 'max': Maximum tone value
        - 'section_coords': List of (row, col, tone) tuples for all sections
    """
    h, w = gray_img.shape
    section_h = h // grid_rows
    section_w = w // grid_cols
    
    # Create arrays to store average tones
    average_tones = np.zeros((grid_rows, grid_cols))
    section_coords = []
    
    # Compute average tone for each section
    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * section_h
            y_end = (i + 1) * section_h if i < grid_rows - 1 else h
            x_start = j * section_w
            x_end = (j + 1) * section_w if j < grid_cols - 1 else w
            
            section = gray_img[y_start:y_end, x_start:x_end]
            avg_tone = np.mean(section)
            average_tones[i, j] = avg_tone
            section_coords.append((i, j, avg_tone))
    
    # Compute statistics
    stats = {
        'mean': np.mean(average_tones),
        'std': np.std(average_tones),
        'min': np.min(average_tones),
        'max': np.max(average_tones),
        'section_coords': section_coords
    }
    
    # Create full-size image if requested
    if return_full_image:
        tonal_img = np.zeros_like(gray_img)
        for i in range(grid_rows):
            for j in range(grid_cols):
                y_start = i * section_h
                y_end = (i + 1) * section_h if i < grid_rows - 1 else h
                x_start = j * section_w
                x_end = (j + 1) * section_w if j < grid_cols - 1 else w
                tonal_img[y_start:y_end, x_start:x_end] = average_tones[i, j]
    else:
        tonal_img = average_tones
    
    return tonal_img, average_tones, stats

