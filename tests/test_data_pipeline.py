"""
Tests for data pipeline: image readability, annotation parsing, coordinate scaling, and leakage prevention.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.data.annotations import parse_annotation_file, scale_boxes
from src.data.dataset import LunarCraterDataset, collate_fn
from src.data.splitting import create_deterministic_splits, extract_group_id, verify_no_leakage
from src.utils.validation import safe_path, validate_bounding_box, validate_image_file


@pytest.fixture
def test_fixture_dir(tmp_path: Path) -> Path:
    """Create isolated test directory with synthetic test fixtures."""
    d = tmp_path / "test_data"
    d.mkdir()

    # Create synthetic test image (800x800 grayscale)
    img_arr = np.full((800, 800), 128, dtype=np.uint8)
    # Draw simulated crater gradient
    img = Image.fromarray(img_arr)
    img.save(d / "M115143943_test_tile.png")

    # Create synthetic annotation file in LRO NAC format: ID,X,Y,W,H,C_X,CY
    lines = [
        "ID,X,Y,W,H,C_X,CY\n",
        "1,100.0,150.0,20.0,20.0,110.0,160.0\n",
        "2,400.0,450.0,40.0,40.0,420.0,470.0\n",
        "3,500.0,500.0,3.0,3.0,501.5,501.5\n",  # Too small (< 6px diameter), should be rejected
    ]
    with open(d / "M115143943_test_tile.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    return d


def test_annotation_parsing(test_fixture_dir: Path):
    """Test parsing authentic LRO crater annotation text format."""
    txt_path = test_fixture_dir / "M115143943_test_tile.txt"
    craters, rejects = parse_annotation_file(txt_path, image_width=800, image_height=800, min_diameter_px=6.0)

    assert len(craters) == 2  # Crater 1 and 2 valid
    assert len(rejects) == 1  # Crater 3 rejected due to size < 6px
    assert craters[0].crater_id == 1
    assert craters[0].xmin == 100.0
    assert craters[0].xmax == 120.0
    assert craters[0].width == 20.0


def test_scale_boxes():
    """Test rescaling bounding boxes from original size (800x800) to target size (320x320)."""
    raw_boxes = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
    scaled = scale_boxes(raw_boxes, orig_size=(800, 800), target_size=(320, 320))

    # Scale factor = 320 / 800 = 0.4
    expected = np.array([[40.0, 80.0, 120.0, 160.0]], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, rtol=1e-5)


def test_image_validation(test_fixture_dir: Path):
    """Test image validation and corruption detection."""
    valid_img = test_fixture_dir / "M115143943_test_tile.png"
    is_ok, err, size = validate_image_file(valid_img)
    assert is_ok is True
    assert size == (800, 800)

    # Test corrupt file
    corrupt_file = test_fixture_dir / "corrupt.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_PNG_FILE_HEADER")
    is_ok, err, size = validate_image_file(corrupt_file)
    assert is_ok is False
    assert "Corrupted or invalid image" in err


def test_bounding_box_validator():
    """Test boundary checking on individual boxes."""
    # Valid box
    ok, err = validate_bounding_box([10, 20, 50, 60], 100, 100)
    assert ok is True

    # Inverted box (xmin >= xmax)
    ok, err = validate_bounding_box([50, 20, 10, 60], 100, 100)
    assert ok is False

    # NaN box
    ok, err = validate_bounding_box([float("nan"), 20, 50, 60], 100, 100)
    assert ok is False


def test_safe_path_traversal():
    """Test path traversal security safeguard."""
    base = Path("C:/data/lunar").resolve() if Path("C:/").exists() else Path("/data/lunar").resolve()
    # Safe subpath
    p = safe_path(base, "tiles/img1.png")
    assert str(p).startswith(str(base))

    # Traversal attempt
    with pytest.raises(ValueError, match="Security Violation"):
        safe_path(base, "../../../secret/credentials.txt")


def test_deterministic_splitting_and_leakage_prevention(tmp_path: Path):
    """Test that grouped splitting partitions tiles by parent observation ID without leakage."""
    records = [
        {"image_name": "M115143943_tile_01.png", "annotation_name": "M115143943_tile_01.txt"},
        {"image_name": "M115143943_tile_02.png", "annotation_name": "M115143943_tile_02.txt"},
        {"image_name": "M1183658592_tile_01.png", "annotation_name": "M1183658592_tile_01.txt"},
        {"image_name": "M1229577647_tile_01.png", "annotation_name": "M1229577647_tile_01.txt"},
        {"image_name": "M130030018_tile_01.png", "annotation_name": "M130030018_tile_01.txt"},
    ]

    splits = create_deterministic_splits(records, output_splits_dir=tmp_path, seed=42)

    # Check that all records are allocated
    total_split_records = sum(len(v) for v in splits.values())
    assert total_split_records == len(records)

    # Verify no leakage
    verify_no_leakage(splits)

    # Artificially introduce leakage and ensure error is raised
    leaked_splits = {
        "train": [records[0]],
        "val": [records[1]],  # Shares parent M115143943 with train!
        "test": [],
    }
    with pytest.raises(ValueError, match="Leakage Detected"):
        verify_no_leakage(leaked_splits)


def test_lunar_crater_dataset(test_fixture_dir: Path):
    """Test PyTorch LunarCraterDataset indexing and collation."""
    records = [{
        "image_name": "M115143943_test_tile.png",
        "annotation_name": "M115143943_test_tile.txt",
        "image_path": "M115143943_test_tile.png",
    }]

    ds = LunarCraterDataset(records, base_dir=test_fixture_dir, target_size=(320, 320))
    assert len(ds) == 1

    img_tensor, target = ds[0]
    assert img_tensor.shape == (3, 320, 320)
    assert target["boxes"].shape == (2, 4)
    assert len(target["labels"]) == 2

    # Test collate_fn
    batch = collate_fn([(img_tensor, target)])
    imgs, targets = batch
    assert len(imgs) == 1
    assert len(targets) == 1
