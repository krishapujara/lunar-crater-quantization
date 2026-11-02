"""
Cryptographic checksum and integrity validation utilities.
Ensures scientific provenance and tamper-detection for datasets and model weights.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hash of a file in streaming chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum computation: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute the SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_dict_sha256(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def verify_file_checksum(file_path: Union[str, Path], expected_sha256: str) -> bool:
    """Verify that a file's SHA-256 matches expected hash."""
    actual_hash = compute_sha256(file_path)
    return actual_hash.lower() == expected_sha256.lower()
