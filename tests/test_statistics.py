import pytest
import numpy as np
from pathlib import Path
from src.processing.statistics import (
    pairwise_distances, 
    group_embeddings_by_video, 
    filter_active_embeddings, 
    analyze_video_formats,
    evaluate_theta_candidates
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



def test_theta_evaluation_counts_and_coverage():
    groups = {
        "duplicates": [
            ("a.jpg", np.array([1.0, 0.0])),
            ("b.jpg", np.array([1.0, 0.0])),
        ],
        "different": [
            ("c.jpg", np.array([1.0, 0.0])),
            ("d.jpg", np.array([0.0, 1.0])),
        ],
        "single": [
            ("e.jpg", np.array([-1.0, 0.0])),
        ],
    }
    original = {
        name: [vector.copy() for _, vector in frames]
        for name, frames in groups.items()
    }

    result = evaluate_theta_candidates(
        groups, [0.0, 0.5, 1.5],
    )

    assert result["total_frames"].tolist() == [5, 5, 5]
    assert result["kept_frames"].tolist() == [5, 4, 3]
    assert result["kept_fraction"].tolist() == pytest.approx(
        [1.0, 0.8, 0.6],
    )
    assert result["coverage_p95"].tolist() == pytest.approx(
        [0.0, 0.0, 0.8],
    )
    assert result["coverage_max"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0],
    )

    for name, frames in groups.items():
        for (_, vector), expected in zip(frames, original[name]):
            np.testing.assert_array_equal(vector, expected)


def test_theta_evaluation_keeps_each_video_represented():
    groups = {
        "first": [
            ("a.jpg", np.array([1.0, 0.0])),
        ],
        "second": [
            ("b.jpg", np.array([1.0, 0.0])),
        ],
    }

    result = evaluate_theta_candidates(groups, [0.5])

    assert result.loc[0, "kept_frames"] == 2
    assert result.loc[0, "coverage_max"] == 0.0


@pytest.mark.parametrize("theta", [0.1, 1.0, 1.1])
def test_theta_evaluation_matches_selector(theta):
    from src.processing.clustering import k_center_greedy_threshold

    matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    groups = {
        "demo": [
            (str(i), vector)
            for i, vector in enumerate(matrix)
        ],
    }

    result = evaluate_theta_candidates(groups, [theta])

    assert result.loc[0, "kept_frames"] == len(
        k_center_greedy_threshold(matrix, theta),
    )


@pytest.mark.parametrize("thetas", [
    [],
    [-0.1],
    [np.nan],
    [np.inf],
    [[0.1]],
])
def test_theta_evaluation_rejects_invalid_thresholds(thetas):
    with pytest.raises(ValueError, match="theta"):
        evaluate_theta_candidates({}, thetas)


def test_theta_evaluation_rejects_empty_groups():
    with pytest.raises(ValueError, match="No frames"):
        evaluate_theta_candidates(
            {"empty": []},
            [0.1],
        )


@pytest.mark.parametrize("vector", [
    np.array([np.nan, 0.0]),
    np.array([2.0, 0.0]),
])
def test_theta_evaluation_rejects_invalid_embeddings(vector):
    with pytest.raises(ValueError, match="L2-normalized"):
        evaluate_theta_candidates(
            {"demo": [("a.jpg", vector)]},
            [0.1],
        )