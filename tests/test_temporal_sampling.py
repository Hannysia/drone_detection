import pandas as pd
import pytest

from src.processing.sampling import (
    collect_frame_runs,
    evaluate_negative_sampling,
    select_temporal_indices,
)


def row(
    number,
    *,
    split="val",
    video="wing_rgb_sky_001",
    category="negative",
    excluded_at="",
):
    return {
        "filename": f"{video}_f{number:04d}.jpg",
        "split": split,
        "category": category,
        "excluded_at": excluded_at,
    }


def test_original_frame_gaps_split_runs():
    runs = collect_frame_runs([
        row(number)
        for number in (57, 3, 1, 31, 2, 28, 30, 29)
    ], "negative"
    )

    assert list(runs["start_frame"]) == [1, 28, 57]
    assert list(runs["stop_frame"]) == [3, 31, 57]
    assert list(runs["frame_count"]) == [3, 4, 1]

    kept = [
        names[index]
        for names in runs["filenames"]
        for index in sorted(select_temporal_indices(len(names), 3))
    ]

    assert kept == [
        row(number)["filename"]
        for number in (1, 3, 28, 31, 57)
    ]


def test_split_and_video_boundaries_are_preserved():
    runs = collect_frame_runs([
        row(1),
        row(2),
        row(1, split="test"),
        row(2, video="wing_rgb_sky_002"),
    ], "negative"
    )

    assert len(runs) == 3
    assert runs["frame_count"].sum() == 4
    assert set(zip(runs["split"], runs["video_id"])) == {
        ("val", "wing_rgb_sky_001"),
        ("test", "wing_rgb_sky_001"),
        ("val", "wing_rgb_sky_002"),
    }


def test_train_positive_and_excluded_rows_are_ignored():
    runs = collect_frame_runs([
        row(1, split="train"),
        row(2, category="positive"),
        row(3, excluded_at="embeddings"),
        row(4),
    ], "negative"
    )

    assert list(runs["start_frame"]) == [4]
    assert list(runs["frame_count"]) == [1]


def test_sparse_negatives_are_all_preserved():
    runs = collect_frame_runs([
        row(number) for number in (1, 28, 57)
    ], "negative"
    )

    result = evaluate_negative_sampling(runs, [2, 3, 4])
    val = result.loc[result["split"] == "val"]

    assert list(val["kept_frames"]) == [3, 3, 3]


@pytest.mark.parametrize(
    "n_frames,step,expected",
    [
        (0, 3, set()),
        (1, 4, {0}),
        (2, 4, {0, 1}),
        (6, 3, {0, 3, 5}),
        (6, 1, set(range(6))),
    ],
)
def test_temporal_selection_preserves_endpoints(
    n_frames,
    step,
    expected,
):
    assert select_temporal_indices(n_frames, step) == expected


@pytest.mark.parametrize(
    "n_frames,step",
    [
        (-1, 2),
        (3, 0),
        (3, 1.5),
    ],
)
def test_invalid_parameters_fail(n_frames, step):
    with pytest.raises(ValueError):
        select_temporal_indices(n_frames, step)


def test_duplicate_original_frame_numbers_fail():
    with pytest.raises(ValueError, match="Duplicate frame number"):
        collect_frame_runs([row(1), row(1)], "negative")


def test_input_order_does_not_change_runs():
    rows = [
        row(number)
        for number in (1, 2, 5, 6)
    ]

    pd.testing.assert_frame_equal(
        collect_frame_runs(rows, "negative"),
        collect_frame_runs(rows[::-1], "negative"),
    )


def test_empty_split_is_reported():
    runs = collect_frame_runs([], "negative")
    result = evaluate_negative_sampling(runs, [2])

    assert list(result["split"]) == ["val", "test"]
    assert result["input_frames"].sum() == 0
    assert result["kept_frames"].sum() == 0


def test_evaluation_matches_selected_run_lengths():
    runs = collect_frame_runs([
        row(number)
        for number in (1, 2, 3, 4, 5, 6, 28, 57)
    ], 
        "negative",
    )

    result = evaluate_negative_sampling(runs, [1, 3])
    val = result.loc[result["split"] == "val"]

    assert list(val["kept_frames"]) == [8, 5]
    assert list(val["removed_frames"]) == [0, 3]