import numpy as np
import pandas as pd
import pytest

from src.processing.theta_analysis import (
    cross_video_candidate_pairs,
    evaluate_cross_video_candidates,
    select_cross_video_frames,
    within_video_coverage,
)


def frame(video, number):
    return f"wing_rgb_sky_{video:03d}_f{number:04d}.jpg"


def make_embeddings(angles):
    return {
        name: np.array([np.cos(angle), np.sin(angle)])
        for name, angle in angles.items()
    }


def test_removes_lower_novelty_frame():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0.1,
        frame(2, 1): 0.01,
        frame(2, 2): 1,
    })
    pairs = cross_video_candidate_pairs(embeddings, 0.001)
    kept, log = select_cross_video_frames(embeddings, pairs, 0.001)

    assert frame(1, 1) not in kept
    assert frame(2, 1) in kept
    assert log.iloc[0]["removed_novelty"] < log.iloc[0]["matched_novelty"]


def test_novelty_updates_after_removal():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0.1,
        frame(1, 3): 0.7,
        frame(2, 1): 0.005,
        frame(2, 2): 1.5,
        frame(3, 1): 0.11,
        frame(3, 2): 0.42,
    })
    pairs = cross_video_candidate_pairs(embeddings, 0.001)
    kept, log = select_cross_video_frames(embeddings, pairs, 0.001)

    assert list(log["removed"]) == [frame(1, 1), frame(3, 1)]
    assert frame(1, 2) in kept


def test_last_frame_of_each_video_is_protected():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(2, 1): 0,
        frame(2, 2): 0,
    })
    pairs = cross_video_candidate_pairs(embeddings, 0.01)
    kept, log = select_cross_video_frames(embeddings, pairs, 0.01)

    assert kept == [frame(1, 1), frame(2, 2)]
    assert len(log) == 1


def test_same_video_pairs_are_excluded():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0,
    })
    pairs = cross_video_candidate_pairs(embeddings, 0.1)
    kept, log = select_cross_video_frames(embeddings, pairs, 0.1)

    assert pairs.empty
    assert log.empty
    assert len(kept) == 2


def test_pairs_are_unique_and_match_exhaustive_search():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0.2,
        frame(2, 1): 0.1,
        frame(3, 1): 2,
    })
    actual = cross_video_candidate_pairs(
        embeddings,
        0.03,
        batch_size=2,
    )
    expected = {
        (a, b)
        for a in embeddings
        for b in embeddings
        if a < b
        and a.rsplit("_f", 1)[0] != b.rsplit("_f", 1)[0]
        and 1 - np.clip(embeddings[a] @ embeddings[b], -1, 1) <= 0.03
    }

    assert set(zip(actual.filename_a, actual.filename_b)) == expected
    assert len(actual) == len(expected)


def test_order_is_deterministic():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0,
        frame(2, 1): 0,
        frame(2, 2): 0,
    })
    pairs = cross_video_candidate_pairs(embeddings, 0.01)
    kept, log = select_cross_video_frames(embeddings, pairs, 0.01)

    shuffled_kept, shuffled_log = select_cross_video_frames(
        dict(reversed(list(embeddings.items()))),
        pairs.iloc[::-1],
        0.01,
    )

    assert kept == shuffled_kept
    pd.testing.assert_frame_equal(log, shuffled_log)


def test_coverage_includes_previously_removed_frames():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0.05,
        frame(1, 3): 0.1,
    })
    groups = {"video": list(embeddings.items())}

    before = within_video_coverage(
        groups,
        [frame(1, 1), frame(1, 3)],
    )
    after = within_video_coverage(groups, [frame(1, 3)])

    assert len(after) == 3
    assert after.max() > before.max()

    with pytest.raises(ValueError, match="No retained representative"):
        within_video_coverage(groups, [])


def test_candidate_evaluation_has_noop_baseline():
    embeddings = make_embeddings({
        frame(1, 1): 0,
        frame(1, 2): 0.1,
        frame(2, 1): 0.01,
        frame(2, 2): 1,
    })
    groups = {
        str(video): [
            (name, value)
            for name, value in embeddings.items()
            if name.startswith(f"wing_rgb_sky_{video:03d}_")
        ]
        for video in (1, 2)
    }

    result, selections, decisions = evaluate_cross_video_candidates(
        embeddings,
        groups,
        [0, 0.001],
    )

    assert list(result["removed_frames"]) == [0, 1]
    assert result.iloc[0]["coverage_increase_max"] == 0
    assert len(selections[0.001]) == 3
    assert len(decisions[0.001]) == 1