"""
Deterministic, leakage-free dataset splitting.
Groups lunar image tiles by observation/orbit strip ID to prevent spatial overlap leakage.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..utils.logging import get_logger

logger = get_logger("data_splitting")


def extract_group_id(filename: str) -> str:
    """
    Extract parent orbit/mosaic product ID from filename.
    Example: 'M115143943_mosaic_train_small.png' or 'M115143943RE_cal_echo_2_2.png'
    Both map to parent observation group 'M115143943'.
    """
    stem = Path(filename).stem
    # Match standard LRO NAC product ID pattern: M followed by digits
    match = re.match(r"(M\d+)", stem)
    if match:
        return match.group(1)
    # Fallback to splitting on underscore or dash
    parts = stem.split("_")
    return parts[0] if parts else stem


def create_deterministic_splits(
    records: List[Dict[str, Any]],
    output_splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create leakage-free train, val, and test splits by grouping on parent product IDs.
    All tiles sharing the same orbit/observation group are assigned exclusively to one split.
    """
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    # Group records by parent observation ID
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        group_id = extract_group_id(r["image_name"])
        groups.setdefault(group_id, []).append(r)

    # Sort groups deterministically
    sorted_group_ids = sorted(groups.keys())

    # Hash-based deterministic shuffling using seed
    def group_sort_key(gid: str) -> str:
        return hashlib.sha256(f"{seed}_{gid}".encode("utf-8")).hexdigest()

    shuffled_groups = sorted(sorted_group_ids, key=group_sort_key)

    total_images = len(records)
    target_train = int(total_images * train_ratio)
    target_val = int(total_images * val_ratio)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    group_assignments: Dict[str, str] = {}

    curr_train = 0
    curr_val = 0

    for gid in shuffled_groups:
        items = groups[gid]
        count = len(items)

        if curr_train < target_train or len(splits["train"]) == 0:
            target_split = "train"
            curr_train += count
        elif curr_val < target_val or len(splits["val"]) == 0:
            target_split = "val"
            curr_val += count
        else:
            target_split = "test"

        group_assignments[gid] = target_split
        for it in items:
            it_copy = dict(it)
            it_copy["group_id"] = gid
            it_copy["assigned_split"] = target_split
            splits[target_split].append(it_copy)

    # Verify zero leakage across splits
    verify_no_leakage(splits)

    # Write manifests
    split_manifest = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_images": total_images,
            "total_groups": len(groups),
            "group_assignments": group_assignments,
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }

    manifest_file = output_splits_dir / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info(
        f"Deterministic splits created without leakage: "
        f"Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])} across {len(groups)} groups."
    )
    return splits


def verify_no_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Verify that no image name or group ID appears in more than one split."""
    seen_images: Dict[str, str] = {}
    seen_groups: Dict[str, str] = {}

    for split_name, items in splits.items():
        for item in items:
            img = item["image_name"]
            gid = item.get("group_id", extract_group_id(img))

            if img in seen_images:
                raise ValueError(f"Data Leakage Detected: Image '{img}' appears in both '{seen_images[img]}' and '{split_name}'.")
            seen_images[img] = split_name

            if gid in seen_groups and seen_groups[gid] != split_name:
                raise ValueError(
                    f"Observation Group Leakage Detected: Group '{gid}' assigned to both '{seen_groups[gid]}' and '{split_name}'."
                )
            seen_groups[gid] = split_name
