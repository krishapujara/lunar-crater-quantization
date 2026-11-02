"""
Lunar crater dataset acquisition and provenance tracking.
Downloads authentic LRO NAC imagery and annotations from open science repositories.
"""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from PIL import Image

from ..utils.checksums import compute_sha256
from ..utils.logging import get_logger
from .annotations import parse_annotation_file

logger = get_logger("data_acquisition")

GITHUB_API_BASE = "https://api.github.com/repos/yizuifangxiuyh/Lunar_Crater_Detection_Data/contents"
RAW_URL_BASE = "https://raw.githubusercontent.com/yizuifangxiuyh/Lunar_Crater_Detection_Data/main"


def fetch_file_list(subdir: str) -> List[Dict[str, Any]]:
    """Query GitHub contents API for file names in a repository subdirectory."""
    url = f"{GITHUB_API_BASE}/LRO_DATA/{subdir}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"GitHub API returned status {resp.status_code} for {subdir}. Using fallback file list.")
    except Exception as e:
        logger.warning(f"Error querying GitHub API: {e}. Using fallback file list.")

    # Fallback to known list of genuine files in the repository
    if subdir == "train":
        names = [
            "M115143943_mosaic_train_small.png", "M115143943_mosaic_train_small.txt",
            "M1155395732_mosaic_train_small.png", "M1155395732_mosaic_train_small.txt",
            "M1183658592_mosaic_train_small.png", "M1183658592_mosaic_train_small.txt",
            "M1229577647_mosaic_train_small.png", "M1229577647_mosaic_train_small.txt",
            "M130030018_mosaic_train_small.png", "M130030018_mosaic_train_small.txt",
            "M134743285_mosaic_train_small.png", "M134743285_mosaic_train_small.txt",
        ]
    else:
        names = [
            "M115143943RE_cal_echo_2_2.png", "M115143943RE_cal_echo_2_2.txt",
            "M1183658592LE_cal_echo_23_0.png", "M1183658592LE_cal_echo_23_0.txt",
            "M1229577647LE_cal_echo_14_4.png", "M1229577647LE_cal_echo_14_4.txt",
            "M130030018RE_cal_echo_1_1.png", "M130030018RE_cal_echo_1_1.txt",
        ]
    return [{"name": n, "download_url": f"{RAW_URL_BASE}/LRO_DATA/{subdir}/{n}"} for n in names]


def download_single_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download a single file with retries and temporary file swapping."""
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return True  # Already downloaded

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=25)
            if resp.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                temp_path.replace(dest_path)
                logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size} bytes)")
                return True
            logger.warning(f"HTTP {resp.status_code} downloading {url} (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            logger.warning(f"Error downloading {url} (attempt {attempt+1}/{max_retries}): {e}")
            if temp_path.exists():
                temp_path.unlink()
        time.sleep(1.0)
    return False


def acquire_lunar_dataset(
    output_raw_dir: Path,
    samples_dir: Path,
    limit_images: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Acquire the authentic LRO crater dataset.
    Downloads genuine images and annotations, computes checksums, and creates manifest.
    """
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, Any]] = []
    total_craters = 0
    rejected_reasons_all: List[str] = []

    for split in ["train", "test"]:
        split_dir = output_raw_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        items = fetch_file_list(split)

        # Pair images and annotation text files
        img_items = [it for it in items if it["name"].endswith(".png")]
        if limit_images:
            img_items = img_items[:limit_images]

        for img_it in img_items:
            img_name = img_it["name"]
            base_stem = Path(img_name).stem
            txt_name = f"{base_stem}.txt"

            img_url = img_it.get("download_url") or f"{RAW_URL_BASE}/LRO_DATA/{split}/{img_name}"
            txt_url = f"{RAW_URL_BASE}/LRO_DATA/{split}/{txt_name}"

            local_img = split_dir / img_name
            local_txt = split_dir / txt_name

            # Download pair
            ok_img = download_single_file(img_url, local_img)
            ok_txt = download_single_file(txt_url, local_txt)

            if not (ok_img and ok_txt):
                logger.error(f"Failed to acquire pair for {img_name}")
                continue

            # Validate and parse
            try:
                with Image.open(local_img) as pil_img:
                    w, h = pil_img.size
                    mode = pil_img.mode

                img_hash = compute_sha256(local_img)
                txt_hash = compute_sha256(local_txt)

                craters, rejects = parse_annotation_file(local_txt, w, h)
                rejected_reasons_all.extend(rejects)
                crater_count = len(craters)
                total_craters += crater_count

                rec = {
                    "image_name": img_name,
                    "annotation_name": txt_name,
                    "split_source": split,
                    "image_path": str(local_img.relative_to(output_raw_dir.parent)),
                    "annotation_path": str(local_txt.relative_to(output_raw_dir.parent)),
                    "width": w,
                    "height": h,
                    "channels": 1 if mode == "L" else 3,
                    "image_sha256": img_hash,
                    "annotation_sha256": txt_hash,
                    "crater_count": crater_count,
                    "size_bytes": local_img.stat().st_size,
                }
                manifest_records.append(rec)

                # Copy first couple of images to samples_dir for zero-friction user testing
                if len(manifest_records) <= 3:
                    sample_target = samples_dir / img_name
                    shutil.copy2(local_img, sample_target)

            except Exception as e:
                logger.error(f"Error validating {img_name}: {e}")

    manifest_data = {
        "dataset_name": "LRO_NAC_ChangE4_Craters",
        "provenance": {
            "mission": "Lunar Reconnaissance Orbiter (NASA)",
            "camera": "Narrow Angle Camera (NAC)",
            "resolution": "0.5 meters/pixel panchromatic band",
            "region": "Von Kármán Crater / Chang'e-4 Landing Zone (45°-46° S, 176.4°-178.8° E)",
            "source_repo": "https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data",
            "license": "GNU General Public License v3.0 (GPL-3.0)",
            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "total_images": len(manifest_records),
            "total_craters": total_craters,
            "average_craters_per_image": round(total_craters / max(1, len(manifest_records)), 1),
            "rejected_annotations_count": len(rejected_reasons_all),
        },
        "records": manifest_records,
    }

    manifest_path = output_raw_dir.parent / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Dataset acquisition complete. Manifest written to {manifest_path} ({len(manifest_records)} images, {total_craters} craters).")
    return manifest_data
