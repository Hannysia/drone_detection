from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.clustering import k_center_greedy_threshold


def frame_number(filename):
    return int(Path(filename).stem.rsplit("_f", 1)[1])


def pair_distance_distributions(groups):
    all_pairs = []
    consecutive = []

    for frames in groups.values():
        if len(frames) < 2:
            continue

        numbers = np.array([frame_number(name) for name, _ in frames])
        matrix = np.stack([vector for _, vector in frames])
        distances = 1 - np.clip(matrix @ matrix.T, -1, 1)

        all_pairs.append(distances[np.triu_indices(len(matrix), k=1)])
        consecutive.append(distances.diagonal(1)[np.diff(numbers) == 1])

    if not all_pairs or not any(values.size for values in consecutive):
        raise ValueError("Expected within-video and consecutive frame pairs")

    return {
        "All within-video pairs": np.concatenate(all_pairs),
        "Consecutive frame pairs": np.concatenate(consecutive),
    }


def compare_video_thresholds(groups, lower, higher, *, length_bins, parse_metadata):
    rows = []

    for video_id, frames in groups.items():
        matrix = np.stack([vector for _, vector in frames])
        distances = 1 - np.clip(matrix @ matrix.T, -1, 1)
        np.fill_diagonal(distances, 0)

        low = k_center_greedy_threshold(matrix, lower)
        high = k_center_greedy_threshold(matrix, higher)
        increase = distances[:, high].min(axis=1) - distances[:, low].min(axis=1)
        removed = len(low) - len(high)
        metadata = parse_metadata(video_id)

        rows.append({
            "video_id": video_id,
            "camera": metadata["camera"],
            "background": metadata["background"],
            "original_frames": len(matrix),
            "kept_lower": len(low),
            "kept_higher": len(high),
            "extra_removed": removed,
            "changed": removed > 0,
            "extra_removed_pp": 100 * removed / len(matrix),
            "distance_increase_p95": np.quantile(increase, 0.95),
        })

    result = pd.DataFrame(rows)
    result["length_group"] = pd.cut(result["original_frames"], bins=length_bins)
    return result


def summarize_video_comparison(videos, columns):
    summary = videos.groupby(columns, observed=True).agg(
        videos=("video_id", "size"),
        changed_videos=("changed", "sum"),
        original_frames=("original_frames", "sum"),
        kept_lower=("kept_lower", "sum"),
        kept_higher=("kept_higher", "sum"),
        extra_removed=("extra_removed", "sum"),
        median_extra_removed_pp=("extra_removed_pp", "median"),
        median_distance_increase_p95=("distance_increase_p95", "median"),
    )
    summary["extra_removed_pct"] = (
        100 * summary["extra_removed"] / summary["original_frames"]
    )
    return summary