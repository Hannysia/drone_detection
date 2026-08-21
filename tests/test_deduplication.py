import pytest
import numpy as np
import imagehash
from pathlib import Path
from src.processing.deduplication import (
    group_labels_by_video, 
    group_files_by_video, 
    detect_video_duplicates
)

def test_group_labels_by_video_standard():
    labels = [
        Path("wing_rgb_sky_001_f0001.txt"),
        Path("wing_rgb_sky_001_f0002.txt"),
        Path("delta_thermal_field_002_f0001.txt")
    ]
    
    groups = group_labels_by_video(labels)
    
    assert "wing_rgb_sky_001" in groups
    assert "delta_thermal_field_002" in groups
    assert len(groups["wing_rgb_sky_001"]) == 2
    assert len(groups["delta_thermal_field_002"]) == 1


def test_group_files_by_video_regex():
    files = [
        Path("wing_rgb_sky_001_f0001.jpg"),
        Path("wing_rgb_sky_001_f0002.jpg"),
        Path("delta_thermal_field_002_f0001.jpg")
    ]
    
    groups = group_files_by_video(files)
    
    assert "wing_rgb_sky_001" in groups
    assert "delta_thermal_field_002" in groups
    assert len(groups["wing_rgb_sky_001"]) == 2
    assert len(groups["delta_thermal_field_002"]) == 1


def test_detect_video_duplicates_standard():
    hash_a = imagehash.ImageHash(np.array([[True, False], [False, True]]))
    hash_b = imagehash.ImageHash(np.array([[True, False], [False, True]]))  # Hamming distance to A is 0
    hash_c = imagehash.ImageHash(np.array([[False, True], [True, False]]))  # Hamming distance to A is 4
    
    signatures = {
        "video_1": [hash_a],
        "video_2": [hash_b],
        "video_3": [hash_c]
    }
    
    # Threshold is 1 (matches distance 0, but ignores distance 4)
    duplicates = detect_video_duplicates(signatures, threshold=1)
    
    assert "video_2" in duplicates
    assert "video_3" not in duplicates