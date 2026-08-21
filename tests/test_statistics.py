import pytest
import numpy as np
from pathlib import Path
from src.processing.statistics import (
    pairwise_distances, 
    group_embeddings_by_video, 
    filter_active_embeddings, 
    analyze_video_formats
)

def test_pairwise_distances_orthogonal():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    dists = pairwise_distances(embs)
    
    assert dists == pytest.approx([1.0])


def test_pairwise_distances_identical():
    embs = np.array([[1.0, 0.0], [1.0, 0.0]])
    
    dists = pairwise_distances(embs)
    
    assert dists == pytest.approx([0.0])


def test_group_embeddings_by_video_standard():
    embs_dict = {
        "wing_rgb_sky_001_f0001.jpg": np.array([1, 2]),
        "wing_rgb_sky_001_f0002.jpg": np.array([3, 4]),
        "delta_thermal_field_002_f0001.jpg": np.array([5, 6]),
    }
    
    groups = group_embeddings_by_video(embs_dict)
    
    assert "wing_rgb_sky_001" in groups
    assert "delta_thermal_field_002" in groups
    assert len(groups["wing_rgb_sky_001"]) == 2
    assert len(groups["delta_thermal_field_002"]) == 1


def test_filter_active_embeddings_standard():
    embs_dict = {
        "frame_1.jpg": np.array([0.1, 0.2]),
        "frame_2.jpg": np.array([0.3, 0.4]),
        "frame_3.jpg": np.array([0.5, 0.6]),
    }
    surviving_names = {"frame_1.jpg", "frame_3.jpg"}
    
    filtered = filter_active_embeddings(embs_dict, surviving_names)
    
    assert len(filtered) == 2
    assert "frame_1.jpg" in filtered
    assert "frame_3.jpg" in filtered
    assert "frame_2.jpg" not in filtered


def test_analyze_video_formats_mock_filesystem(tmp_path):
    sub_dir = tmp_path / "subfolder"
    sub_dir.mkdir()
    (sub_dir / "video1.mp4").write_text("")
    (sub_dir / "video2.mov").write_text("")
    (sub_dir / "image1.jpg").write_text("")
    (tmp_path / "raw_file").write_text("")  # File with no extension

    ext_counter, folders, files = analyze_video_formats(tmp_path)
    
    assert folders == 2
    assert files == 4
    assert ext_counter[".mp4"] == 1
    assert ext_counter[".mov"] == 1
    assert ext_counter[".jpg"] == 1
    assert ext_counter["no_extension"] == 1