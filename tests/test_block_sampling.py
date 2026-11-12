import pandas as pd
import pytest

from src.processing.sampling import (
    collect_frame_runs,
    evaluate_block_sampling,
    select_block_indices,
)


def negative_row(number, split="val"):
    return {
        "filename": f"wing_rgb_sky_001_f{number:04d}.jpg",
        "split": split,
        "category": "negative",
        "excluded_at": "",
    }


@pytest.mark.parametrize(
    "n_frames,block_size,expected",
    [
        (0, 5, set()),
        (1, 5, {0}),
        (2, 5, {0}),
        (3, 5, {1}),
        (5, 5, {2}),
        (6, 5, {2, 5}),
        (10, 5, {2, 7}),
        (20, 10, {4, 14}),
    ],
)
def test_block_centers(n_frames, block_size, expected):
    assert select_block_indices(n_frames, block_size) == expected


def test_every_block_has_exactly_one_representative():
    for n_frames in range(131):
        for block_size in (1, 2, 3, 5, 10):
            selected = select_block_indices(n_frames, block_size)

            assert len(selected) == (
                n_frames + block_size - 1
            ) // block_size

            for start in range(0, n_frames, block_size):
                block = set(range(
                    start,
                    min(start + block_size, n_frames),
                ))
                assert len(selected & block) == 1


@pytest.mark.parametrize(
    "n_frames,block_size",
    [(-1, 5), (5, 0), (5, 2.5)],
)
def test_invalid_parameters_fail(n_frames, block_size):
    with pytest.raises(ValueError):
        select_block_indices(n_frames, block_size)


def test_sparse_runs_and_split_boundaries():
    rows = [
        negative_row(number)
        for number in (1, 2, 3, 4, 5, 28, 57)
    ]
    rows += [
        negative_row(number, "test")
        for number in range(1, 21)
    ]

    runs = collect_frame_runs(rows, "negative")
    result = evaluate_block_sampling(runs, [5, 10])

    assert set(result["split"]) == {"val"}
    assert list(result["input_frames"]) == [7, 7]
    assert list(result["kept_frames"]) == [3, 3]
    assert list(result["runs"]) == [3, 3]


def test_input_order_and_duplicate_candidates_do_not_change_results():
    rows = [
        negative_row(number)
        for number in range(1, 14)
    ]

    first = evaluate_block_sampling(
        collect_frame_runs(rows, "negative"),
        [5, 10],
    )
    second = evaluate_block_sampling(
        collect_frame_runs(rows[::-1], "negative"),
        [10, 5, 5],
    )

    pd.testing.assert_frame_equal(first, second)


def test_missing_split_fails():
    runs = collect_frame_runs([negative_row(1)], "negative")

    with pytest.raises(ValueError, match="No active runs"):
        evaluate_block_sampling(
            runs,
            [5],
            split="test",
        )


        