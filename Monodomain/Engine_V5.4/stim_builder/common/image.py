"""
Image processing utilities shared between builders.
"""

from PIL import Image
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set

from .utils import is_background_color


def load_image(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load an image file and convert to numpy array.

    Args:
        path: Path to image file

    Returns:
        Tuple of (image_array, (width, height))
    """
    with Image.open(Path(path)) as img:
        if img.mode == 'RGBA':
            image_array = np.array(img)
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            image_array = np.array(img)
        else:
            image_array = np.array(img)

        return image_array, img.size


def detect_colors(
    image_array: np.ndarray,
    auto_detect_background: bool = True
) -> Dict[Tuple[int, ...], dict]:
    """
    Detect all distinct colors in an image.

    Args:
        image_array: numpy array of image pixels
        auto_detect_background: If True, mark white/transparent as background

    Returns:
        Dict mapping color tuples to info dicts with keys:
        - pixel_count: int
        - is_background: bool
    """
    pixels = image_array.reshape(-1, image_array.shape[-1])
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    color_info = {}
    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(color.tolist())
        is_bg = is_background_color(color_tuple) if auto_detect_background else False

        color_info[color_tuple] = {
            'pixel_count': int(count),
            'is_background': is_bg,
        }

    return color_info


def threshold_transparency(
    image_array: np.ndarray,
    alpha_threshold: int = 128
) -> np.ndarray:
    """
    Binarize alpha channel to remove anti-aliasing artifacts.

    For RGBA images:
    - Pixels with alpha >= threshold become fully opaque (A=255)
    - Pixels below threshold become fully transparent (A=0) with RGB=(0,0,0)

    Args:
        image_array: RGBA image array
        alpha_threshold: Threshold value (0-255)

    Returns:
        Processed image array
    """
    result = image_array.copy()

    if result.shape[-1] == 4:
        alpha = result[:, :, 3]
        is_transparent = alpha < alpha_threshold

        # Set transparent pixels to (0,0,0,0)
        result[is_transparent, 0] = 0
        result[is_transparent, 1] = 0
        result[is_transparent, 2] = 0
        result[is_transparent, 3] = 0

        # Set opaque pixels alpha to 255
        result[~is_transparent, 3] = 255

    return result


def apply_mode_filter(
    image_array: np.ndarray,
    small_colors: Set[Tuple[int, ...]],
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply mode filter to pixels belonging to small color groups.

    For each small-group pixel, replace with the most common color
    among its neighbors (excluding other small-group colors).

    Args:
        image_array: Image array
        small_colors: Set of color tuples to be filtered
        kernel_size: Size of the neighborhood (3, 5, 7, etc.)

    Returns:
        Filtered image array
    """
    result = image_array.copy()
    h, w = image_array.shape[:2]
    pad = kernel_size // 2

    # Find positions of small-group pixels
    small_positions = []
    for y in range(h):
        for x in range(w):
            color = tuple(image_array[y, x].tolist())
            if color in small_colors:
                small_positions.append((y, x))

    # For each small-group pixel, find mode of neighbors
    for y, x in small_positions:
        y_min = max(0, y - pad)
        y_max = min(h, y + pad + 1)
        x_min = max(0, x - pad)
        x_max = min(w, x + pad + 1)

        neighbor_colors = []
        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if ny == y and nx == x:
                    continue
                color = tuple(image_array[ny, nx].tolist())
                if color not in small_colors:
                    neighbor_colors.append(color)

        if neighbor_colors:
            mode_color = Counter(neighbor_colors).most_common(1)[0][0]
            result[y, x] = mode_color

    return result


def filter_small_groups(
    image_array: np.ndarray,
    color_info: Dict[Tuple[int, ...], dict],
    min_percent: float = 0.1
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], dict]]:
    """
    Filter out small color groups using adaptive mode filtering.

    Tries 3x3, 5x5, 7x7 kernels until small groups are eliminated.
    Always operates on original image to avoid compounding errors.

    Args:
        image_array: Image array
        color_info: Dict from detect_colors()
        min_percent: Groups below this percent of tissue pixels are "small"

    Returns:
        Tuple of (filtered_image_array, new_color_info)
    """
    original_array = image_array.copy()

    # Calculate threshold
    total_tissue_pixels = sum(
        info['pixel_count'] for info in color_info.values()
        if not info['is_background']
    )
    min_pixels = int(total_tissue_pixels * min_percent / 100)

    def get_small_colors(info: Dict) -> Set[Tuple[int, ...]]:
        return {
            color for color, data in info.items()
            if not data['is_background'] and data['pixel_count'] < min_pixels
        }

    small_colors = get_small_colors(color_info)
    if not small_colors:
        return image_array, color_info

    best_result = None
    best_small_count = float('inf')

    for kernel_size in [3, 5, 7]:
        # Apply mode filter on original
        filtered = apply_mode_filter(original_array, small_colors, kernel_size)
        new_info = detect_colors(filtered)
        new_small = get_small_colors(new_info)

        if not new_small:
            print(f"filter_small_groups: {kernel_size}x{kernel_size} eliminated all small groups")
            return filtered, new_info

        if len(new_small) < best_small_count:
            best_small_count = len(new_small)
            best_result = (filtered, new_info, kernel_size)

    if best_result:
        filtered, new_info, kernel = best_result
        print(f"filter_small_groups: using {kernel}x{kernel} (reduced to {best_small_count} small groups)")
        return filtered, new_info

    return image_array, color_info
