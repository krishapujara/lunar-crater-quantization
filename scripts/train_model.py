"""
CLI Script: Train SSDLite320 Lunar Crater Object Detector.
Usage: python scripts/train_model.py [--epochs N] [--batch-size B] [--device cpu|cuda]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.training.train import train_crater_detector
from src.utils.logging import get_logger

logger = get_logger("cli_train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSDLite320 Lunar Crater Detector")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    cfg = load_config()
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.device:
        cfg.training.device = args.device

    split_manifest_file = cfg.paths.splits_dir / "split_manifest.json"
    if not split_manifest_file.is_file():
        logger.error(
            f"Split manifest not found at {split_manifest_file}. "
            "Please run 'python scripts/download_data.py' first."
        )
        sys.exit(1)

    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_records = split_data["splits"]["train"]
    val_records = split_data["splits"]["val"]

    logger.info(f"Loaded {len(train_records)} train records and {len(val_records)} validation records.")
    model, best_ckpt_path, summary = train_crater_detector(
        cfg=cfg,
        train_records=train_records,
        val_records=val_records,
    )

    print(f"\n--- Training Complete ---")
    print(f"Best Validation AP@0.50: {summary['best_val_ap50']}")
    print(f"Best Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
