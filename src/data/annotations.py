"""
Lunar crater annotation parsing, conversion, and validation.
Parses authentic LRO NAC format: ID,X,Y,W,H,C_X,CY and transforms to detection boxes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class CraterRecord:
    crater_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    center_x: float
    center_y: float
    width: float
    height: float
    diameter: float

    @property
    def box_xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def box_xywh(self) -> List[float]:
        return [self.xmin, self.ymin, self.width, self.height]


def parse_annotation_file(
    annotation_path: Union[str, Path],
    image_width: int,
    image_height: int,
    min_diameter_px: float = 6.0,
) -> Tuple[List[CraterRecord], List[str]]:
    """
    Parse an authentic LRO crater annotation file (.txt).
    Expected line format: ID,X,Y,W,H,C_X,CY (or with whitespace).
    Returns: (valid_craters, rejected_reasons)
    """
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    valid_craters: List[CraterRecord] = []
    rejected_reasons: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("id"):
            continue  # Skip header or empty line

        # Split on comma or whitespace
        parts = [p.strip() for p in (line.split(",") if "," in line else line.split())]
        if len(parts) < 5:
            rejected_reasons.append(f"Line {line_idx+1}: Insufficient tokens ({len(parts)} < 5)")
            continue

        try:
            crater_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Optional center coordinates
            cx = float(parts[5]) if len(parts) > 5 else (x + w / 2.0)
            cy = float(parts[6]) if len(parts) > 6 else (y + h / 2.0)

            diameter = max(w, h)

            if diameter < min_diameter_px:
                rejected_reasons.append(f"Line {line_idx+1} (ID {crater_id}): Diameter {diameter:.1f}px < threshold {min_diameter_px}px")
                continue

            xmin = max(0.0, x)
            ymin = max(0.0, y)
            xmax = min(float(image_width), x + w)
            ymax = min(float(image_height), y + h)

            # Check validity after clipping
            if xmax <= xmin or ymax <= ymin:
                rejected_reasons.append(f"Line {line_idx+1} (ID {crater_id}): Invalid bounds after clipping [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")
                continue

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w < 2.0 or box_h < 2.0:
                rejected_reasons.append(f"Line {line_idx+1} (ID {crater_id}): Box too small ({box_w:.1f}x{box_h:.1f})")
                continue

            valid_craters.append(
                CraterRecord(
                    crater_id=crater_id,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    center_x=cx,
                    center_y=cy,
                    width=box_w,
                    height=box_h,
                    diameter=diameter,
                )
            )
        except ValueError as e:
            rejected_reasons.append(f"Line {line_idx+1}: Parse error '{str(e)}'")

    return valid_craters, rejected_reasons


def scale_boxes(
    boxes: np.ndarray,
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale bounding boxes [xmin, ymin, xmax, ymax] from orig_size (W, H) to target_size (W, H).
    """
    if len(boxes) == 0:
        return boxes.copy()

    orig_w, orig_h = orig_size
    target_w, target_h = target_size

    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)

    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y

    # Clip to target boundaries
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, target_w)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, target_h)
    return scaled
