import pytest
import numpy as np
from pathlib import Path
from src.processing.dataset_statistics import analyze_video_formats
from src.processing.embedding_statistics import (
    evaluate_theta_candidates,
    filter_active_embeddings,
    group_embeddings_by_video,
    pairwise_distances,
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



def test_get_dataset_durations_metadata_raises_on_broken_video(tmp_path, monkeypatch):
    from src.processing import dataset_statistics

    video = tmp_path / "broken.mp4"
    video.write_bytes(b"bad")

    monkeypatch.setattr(dataset_statistics, "get_video_metadata", lambda _: (0, 0, 0.0, 0))
    with pytest.raises(ValueError, match="Invalid video metadata"):
        dataset_statistics.get_dataset_durations_metadata(tmp_path, [".mp4"])

    def fail_meta(_):
        raise OSError("cannot open")

    monkeypatch.setattr(dataset_statistics, "get_video_metadata", fail_meta)
    with pytest.raises(RuntimeError, match="Failed to read video metadata"):
        dataset_statistics.get_dataset_durations_metadata(tmp_path, [".mp4"])


def test_gather_all_splits_geometry_raises_on_corrupt_image_or_label(tmp_path):
    from PIL import Image
    from src.processing.dataset_statistics import gather_all_splits_geometry

    pos_dir = tmp_path / "train" / "RAW_POSITIVES"
    pos_dir.mkdir(parents=True)
    img_path = pos_dir / "wing_rgb_sky_001_f0001.jpg"
    lbl_path = img_path.with_suffix(".txt")

    img_path.write_bytes(b"not an image")
    lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreadable image header"):
        gather_all_splits_geometry(tmp_path, ["train"])

    Image.new("RGB", (32, 32)).save(img_path)
    lbl_path.write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5 YOLO fields"):
        gather_all_splits_geometry(tmp_path, ["train"])

    lbl_path.write_text("0 a b c d\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid box coordinates"):
        gather_all_splits_geometry(tmp_path, ["train"])


def test_load_and_verify_embeddings_raises_on_error_or_empty(tmp_path, monkeypatch):
    from src.processing import embedding_statistics

    def fail_download(**kwargs):
        raise OSError("network error")

    monkeypatch.setattr(embedding_statistics, "hf_hub_download", fail_download)
    with pytest.raises(OSError, match="network error"):
        embedding_statistics.load_and_verify_embeddings("repo/id", "embs.pt")

    dummy = tmp_path / "embs.pt"
    dummy.write_bytes(b"pt")
    monkeypatch.setattr(embedding_statistics, "hf_hub_download", lambda **kwargs: str(dummy))
    monkeypatch.setattr(embedding_statistics, "load_embedding_file", lambda _: {})

    with pytest.raises(ValueError, match="empty"):
        embedding_statistics.load_and_verify_embeddings("repo/id", "embs.pt")