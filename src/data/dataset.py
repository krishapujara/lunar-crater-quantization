"""
PyTorch Dataset and DataLoader collation for Lunar Crater Detection.
Converts grayscale LRO NAC images to standardized 3-channel tensors and rescales bounding boxes.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import parse_annotation_file, scale_boxes


class LunarCraterDataset(Dataset):
    """
    Dataset for lunar crater detection with SSDLite320 / MobileNetV3.
    Loads raw LRO NAC tiles and scales coordinates to target resolution (default 320x320).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        base_dir: Path,
        target_size: Tuple[int, int] = (320, 320),
        min_diameter_px: float = 6.0,
        transforms: Optional[Callable] = None,
    ):
        self.records = records
        self.base_dir = Path(base_dir)
        self.target_size = target_size
        self.min_diameter_px = min_diameter_px
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        img_rel = rec.get("image_path", rec["image_name"])
        txt_rel = rec.get("annotation_path", rec.get("annotation_name", ""))

        img_path = self.base_dir / img_rel
        if not img_path.is_file():
            # Try searching in raw subdirectories
            for candidate in self.base_dir.rglob(Path(img_rel).name):
                if candidate.is_file():
                    img_path = candidate
                    break

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        txt_path = img_path.with_suffix(".txt")

        # Load image with PIL and convert to RGB (SSDLite backbone expects 3 channels)
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            orig_w, orig_h = pil_img.size
            resized_img = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Parse ground-truth craters
        craters, _ = parse_annotation_file(
            txt_path,
            image_width=orig_w,
            image_height=orig_h,
            min_diameter_px=self.min_diameter_px,
        )

        if len(craters) > 0:
            raw_boxes = np.array([c.box_xyxy for c in craters], dtype=np.float32)
            scaled = scale_boxes(raw_boxes, orig_size=(orig_w, orig_h), target_size=self.target_size)
            boxes_tensor = torch.as_tensor(scaled, dtype=torch.float32)
            labels_tensor = torch.ones((len(craters),), dtype=torch.int64)  # 1: crater
            areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(craters),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Convert image to [0.0, 1.0] float tensor (C, H, W)
        img_arr = np.array(resized_img, dtype=np.float32) / 255.0
        # Transpose from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "image_name": rec["image_name"],
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, Any], ...]]:
    """
    Collate function for Torchvision detection models.
    Returns: (tuple of image tensors, tuple of target dictionaries)
    """
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)
