from pathlib import Path
import numpy as np
import pandas as pd

def select_indices(n_frames: int, every_n: int, cap: int | None) -> set[int]:
    """
    Returns a set of frame indices that should be KEPT.

    Logic:
    1. If cap is set and n_frames > cap — uniformly sample 'cap' indices.
    2. If every_n is set — keep every every_n-th frame.
    3. Otherwise, keep all frames.
    """
    if n_frames == 0:
        return set()

    if cap is not None and n_frames > cap:
        indices = np.linspace(0, n_frames - 1, cap, dtype=int)
        return set(indices.tolist())

    if every_n is not None and every_n > 1:
        return {i for i in range(n_frames) if i % every_n == 0}

    return set(range(n_frames))


def collect_frame_runs(rows: list[dict[str, str]], category: str) -> pd.DataFrame:

    if category not in ("positive", "negative"):
        raise ValueError("Expected positive or negative")

    groups = {}

    for row in rows:
        if (
            row["split"] not in ("val", "test")
            or row["category"] != category
            or row.get("excluded_at", "")
        ):
            continue

        video_id, separator, number = (
            Path(row["filename"]).stem.rpartition("_f")
        )

        if (
            not video_id
            or not separator
            or not number.isdecimal()
            or int(number) < 1
        ):
            raise ValueError(
                f"Invalid frame filename: {row['filename']}"
            )

        key = (row["split"], video_id)
        groups.setdefault(key, []).append(
            (int(number), row["filename"])
        )

    records = []

    for (split, video_id), frames in sorted(groups.items()):
        frames.sort(key=lambda item: item[0])
        numbers = np.array([number for number, _ in frames])

        if len(set(numbers)) != len(numbers):
            raise ValueError(
                f"Duplicate frame number in {split}/{video_id}"
            )

        boundaries = np.flatnonzero(np.diff(numbers) != 1) + 1

        for indices in np.split(np.arange(len(frames)), boundaries):
            records.append({
                "split": split,
                "video_id": video_id,
                "start_frame": int(numbers[indices[0]]),
                "stop_frame": int(numbers[indices[-1]]),
                "frame_count": len(indices),
                "filenames": tuple(
                    frames[index][1] for index in indices
                ),
            })

    return pd.DataFrame(
        records,
        columns=[
            "split",
            "video_id",
            "start_frame",
            "stop_frame",
            "frame_count",
            "filenames",
        ],
    )


def select_temporal_indices(n_frames: int, step: int) -> set[int]:
    if not isinstance(n_frames, int) or n_frames < 0:
        raise ValueError("n_frames must be a nonnegative integer")

    if not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")

    selected = select_indices(n_frames, step, None)

    if n_frames:
        selected.add(n_frames - 1)

    return selected


def evaluate_negative_sampling(runs: pd.DataFrame, steps: list[int]) -> pd.DataFrame:
    if not steps:
        raise ValueError("Provide at least one sampling step")

    for step in steps:
        select_temporal_indices(0, step)

    records = []

    for split in ("val", "test"):
        split_runs = runs.loc[runs["split"] == split]
        total = int(split_runs["frame_count"].sum())

        for step in sorted(set(steps)):
            kept = sum(
                len(select_temporal_indices(len(names), step))
                for names in split_runs["filenames"]
            )

            records.append({
                "split": split,
                "step": step,
                "videos": split_runs["video_id"].nunique(),
                "runs": len(split_runs),
                "input_frames": total,
                "kept_frames": kept,
                "removed_frames": total - kept,
                "removed_pct": (
                    100 * (total - kept) / total if total else 0.0
                ),
            })

    return pd.DataFrame(records)



def select_block_indices(n_frames: int, block_size: int) -> set[int]:
    if not isinstance(n_frames, int) or n_frames < 0:
        raise ValueError("n_frames must be a nonnegative integer")

    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")

    selected = set()

    for start in range(0, n_frames, block_size):
        stop = min(start + block_size, n_frames)
        selected.add((start + stop - 1) // 2)

    return selected


def evaluate_block_sampling(
    runs: pd.DataFrame,
    block_sizes: list[int],
    *,
    split: str = "val",
) -> pd.DataFrame:
    if split not in ("val", "test"):
        raise ValueError("Expected val or test")

    if not block_sizes:
        raise ValueError("Provide at least one block size")

    for block_size in block_sizes:
        select_block_indices(0, block_size)

    split_runs = runs.loc[runs["split"] == split]

    if split_runs.empty:
        raise ValueError(f"No active runs in {split}")

    records = []

    for block_size in sorted(set(block_sizes)):
        for video_id, video_runs in split_runs.groupby(
            "video_id",
            sort=True,
        ):
            total = sum(
                len(names) for names in video_runs["filenames"]
            )
            kept = sum(
                len(select_block_indices(len(names), block_size))
                for names in video_runs["filenames"]
            )

            records.append({
                "split": split,
                "video_id": video_id,
                "block_size": block_size,
                "runs": len(video_runs),
                "input_frames": total,
                "kept_frames": kept,
                "removed_frames": total - kept,
                "removed_pct": 100 * (total - kept) / total,
            })

    return pd.DataFrame(records)