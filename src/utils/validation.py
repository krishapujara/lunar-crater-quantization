"""
Input validation, security safeguards, and boundary checking.
Prevents directory traversal, oversized uploads, and corrupt scientific records.
"""

import io
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_IMAGE_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit for uploaded lunar images


def safe_path(base_dir: Union[str, Path], sub_path: Union[str, Path]) -> Path:
    """
    Ensure sub_path resolves inside base_dir to prevent path traversal attacks.
    Raises ValueError if sub_path escapes base_dir.
    """
    base = Path(base_dir).resolve()
    target = (base / sub_path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Security Violation: Target path '{target}' escapes base directory '{base}'.")
    return target


def validate_image_file(file_path_or_bytes: Union[str, Path, bytes], max_size_bytes: int = MAX_IMAGE_FILE_SIZE_BYTES) -> Tuple[bool, str, Tuple[int, int]]:
    """
    Validate that an image file exists, is within size boundaries, and can be decoded.
    Returns: (is_valid, error_message, (width, height))
    """
    try:
        if isinstance(file_path_or_bytes, (str, Path)):
            p = Path(file_path_or_bytes)
            if not p.is_file():
                return False, f"Image file does not exist: {p}", (0, 0)
            if p.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                return False, f"Unsupported image extension '{p.suffix}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}", (0, 0)
            size = p.stat().st_size
            if size > max_size_bytes:
                return False, f"Image file exceeds maximum allowed size ({size} > {max_size_bytes} bytes).", (0, 0)
            with Image.open(p) as img:
                img.verify()
            # Reopen to read dimensions since verify() closes/invalidates the image object
            with Image.open(p) as img:
                return True, "", img.size
        elif isinstance(file_path_or_bytes, bytes):
            if len(file_path_or_bytes) > max_size_bytes:
                return False, f"Image bytes exceed size limit ({len(file_path_or_bytes)} > {max_size_bytes}).", (0, 0)
            stream = io.BytesIO(file_path_or_bytes)
            with Image.open(stream) as img:
                img.verify()
            stream.seek(0)
            with Image.open(stream) as img:
                return True, "", img.size
        else:
            return False, f"Invalid image input type: {type(file_path_or_bytes)}", (0, 0)
    except Exception as e:
        return False, f"Corrupted or invalid image: {str(e)}", (0, 0)


def validate_bounding_box(box: List[float], image_width: int, image_height: int, min_size_px: float = 2.0) -> Tuple[bool, str]:
    """
    Validate a single bounding box [xmin, ymin, xmax, ymax].
    Checks:
    - Exactly 4 coordinates
    - Non-NaN, non-inf
    - xmin < xmax and ymin < ymax
    - width and height >= min_size_px
    - Box overlaps with image boundaries
    """
    if len(box) != 4:
        return False, f"Bounding box must contain 4 coordinates, got {len(box)}."

    xmin, ymin, xmax, ymax = box
    for val in (xmin, ymin, xmax, ymax):
        if not np.isfinite(val):
            return False, "Bounding box contains non-finite values (NaN or Inf)."

    width = xmax - xmin
    height = ymax - ymin

    if width < min_size_px or height < min_size_px:
        return False, f"Box dimensions too small ({width:.1f}x{height:.1f} < min {min_size_px}px)."

    if xmax <= 0 or ymax <= 0 or xmin >= image_width or ymin >= image_height:
        return False, f"Box [{xmin}, {ymin}, {xmax}, {ymax}] is entirely outside image dimensions ({image_width}x{image_height})."

    return True, ""
