import numpy as np
import pandas as pd
import pytest

from src.processing.theta_analysis import nearest_cross_video_frames


def sample_embeddings():
    return {
        "wing_rgb_sky_001_f0001.jpg": np.array([1.0, 0.0]),
        "wing_rgb_sky_001_f0002.jpg": np.array([1.0, 0.0]),
        "wing_rgb_sky_002_f0001.jpg": np.array([0.8, 0.6]),
        "wing_rgb_sky_003_f0001.jpg": np.array([0.0, 1.0]),
    }


def test_excludes_every_frame_of_the_same_video():
    result = nearest_cross_video_frames(sample_embeddings())

    assert (result["video_id"] != result["neighbor_video_id"]).all()
    assert result.iloc[0]["neighbor_filename"] == "wing_rgb_sky_002_f0001.jpg"
    assert result.iloc[0]["distance"] == pytest.approx(0.2)
    assert result.iloc[-1]["distance"] == pytest.approx(0.4)


def test_matches_exhaustive_search():
    embeddings = sample_embeddings()
    result = nearest_cross_video_frames(embeddings, batch_size=2)

    for row in result.itertuples():
        expected = min(
            1 - np.clip(embeddings[row.filename] @ vector, -1, 1)
            for filename, vector in embeddings.items()
            if filename.rsplit("_f", 1)[0] != row.video_id
        )
        assert row.distance == pytest.approx(expected)


def test_order_and_batches_preserve_results():
    embeddings = sample_embeddings()

    original = nearest_cross_video_frames(embeddings, batch_size=1)
    shuffled = nearest_cross_video_frames(
        dict(reversed(list(embeddings.items()))),
        batch_size=3,
    )

    pd.testing.assert_frame_equal(original, shuffled)


def test_identical_frames_from_different_videos_have_zero_distance():
    embeddings = sample_embeddings()
    embeddings["wing_rgb_sky_002_f0001.jpg"] = np.array([1.0, 0.0])

    result = nearest_cross_video_frames(embeddings)

    assert result.iloc[0]["distance"] == 0


def test_one_video_fails():
    with pytest.raises(ValueError, match="at least two videos"):
        nearest_cross_video_frames({
            "wing_rgb_sky_001_f0001.jpg": np.array([1.0, 0.0]),
        })


def test_invalid_vector_fails():
    embeddings = sample_embeddings()
    embeddings["wing_rgb_sky_002_f0001.jpg"] = np.array([2.0, 0.0])

    with pytest.raises(ValueError, match="normalized"):
        nearest_cross_video_frames(embeddings)