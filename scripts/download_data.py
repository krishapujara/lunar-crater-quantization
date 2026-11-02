"""
CLI Script: Download and Prepare Authentic Lunar Crater Dataset.
Usage: python scripts/download_data.py [--limit N]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.acquisition import acquire_lunar_dataset
from src.data.splitting import create_deterministic_splits
from src.data.validation import validate_dataset_directory
from src.utils.logging import get_logger

logger = get_logger("cli_download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and validate LRO NAC Lunar Crater Dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to download for rapid testing")
    args = parser.parse_args()

    cfg = load_config()
    logger.info("Step 1: Downloading genuine Lunar Reconnaissance Orbiter (LRO) crater dataset...")
    manifest = acquire_lunar_dataset(
        output_raw_dir=cfg.paths.raw_data_dir,
        samples_dir=cfg.paths.samples_dir,
        limit_images=args.limit,
    )

    logger.info("Step 2: Validating downloaded images and annotation files...")
    report = validate_dataset_directory(cfg.paths.raw_data_dir)
    print(f"\n--- Validation Summary ---")
    print(f"Total Images Validated: {report.valid_images_count}/{report.total_images_checked}")
    print(f"Total Valid Craters: {report.total_valid_craters}")
    print(f"Total Rejected Annotations: {report.total_rejected_craters}")
    print(f"Corrupt Images: {report.corrupt_images_count}")
    print(f"Missing Annotations: {report.missing_annotations_count}")

    logger.info("Step 3: Creating deterministic, leakage-free train, val, and test splits...")
    splits = create_deterministic_splits(
        records=manifest["records"],
        output_splits_dir=cfg.paths.splits_dir,
        train_ratio=cfg.data.split_ratios["train"],
        val_ratio=cfg.data.split_ratios["val"],
        test_ratio=cfg.data.split_ratios["test"],
        seed=cfg.project.random_seed,
    )

    print(f"\n--- Split Summary ---")
    print(f"Train Tiles: {len(splits['train'])}")
    print(f"Validation Tiles: {len(splits['val'])}")
    print(f"Test Tiles: {len(splits['test'])}")
    print(f"\nDataset acquisition complete! Manifest at: {cfg.paths.data_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
