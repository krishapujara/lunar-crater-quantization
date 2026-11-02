"""
Dataset integrity and annotation validation pipeline.
Detects missing, corrupt, duplicate, and incompatible records before training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

from ..utils.logging import get_logger
from ..utils.validation import validate_image_file
from .annotations import parse_annotation_file

logger = get_logger("data_validation")


@dataclass
class ValidationReport:
    total_images_checked: int = 0
    valid_images_count: int = 0
    corrupt_images_count: int = 0
    missing_annotations_count: int = 0
    total_valid_craters: int = 0
    total_rejected_craters: int = 0
    image_errors: Dict[str, str] = field(default_factory=dict)
    annotation_rejections: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.valid_images_count > 0 and self.corrupt_images_count == 0 and self.missing_annotations_count == 0


def validate_dataset_directory(data_dir: Path) -> ValidationReport:
    """
    Scan a directory containing images and annotations and validate every record.
    """
    report = ValidationReport()
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        logger.error(f"Dataset directory not found: {data_dir}")
        return report

    # Search for all image files
    img_files = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
    report.total_images_checked = len(img_files)

    for img_path in img_files:
        is_ok, err_msg, (w, h) = validate_image_file(img_path)
        if not is_ok:
            report.corrupt_images_count += 1
            report.image_errors[img_path.name] = err_msg
            continue

        # Check annotation counterpart
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.is_file():
            report.missing_annotations_count += 1
            report.image_errors[img_path.name] = "Missing corresponding .txt annotation file."
            continue

        # Parse annotations
        craters, rejects = parse_annotation_file(txt_path, w, h)
        report.total_valid_craters += len(craters)
        if rejects:
            report.total_rejected_craters += len(rejects)
            report.annotation_rejections[txt_path.name] = rejects

        report.valid_images_count += 1

    logger.info(
        f"Validation complete: {report.valid_images_count}/{report.total_images_checked} valid images, "
        f"{report.total_valid_craters} valid craters, {report.total_rejected_craters} rejected boxes."
    )
    return report
