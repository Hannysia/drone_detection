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



def nearest_cross_video_frames(embeddings, *, batch_size=256):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if not embeddings:
        raise ValueError("No embeddings to analyze")

    filenames = sorted(embeddings)
    video_ids = []

    for name in filenames:
        frame_number(name)
        video_ids.append(Path(name).stem.rsplit("_f", 1)[0])

    video_ids = np.array(video_ids)

    if len(set(video_ids)) < 2:
        raise ValueError("Expected frames from at least two videos")

    matrix = np.stack([embeddings[name] for name in filenames])

    if (
        matrix.ndim != 2
        or not matrix.shape[1]
        or not np.issubdtype(matrix.dtype, np.floating)
        or not np.isfinite(matrix).all()
        or not np.allclose(
            np.linalg.norm(matrix, axis=1),
            1,
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise ValueError("Expected finite, L2-normalized embeddings")

    records = []

    for start in range(0, len(matrix), batch_size):
        stop = min(start + batch_size, len(matrix))
        distances = 1 - np.clip(matrix[start:stop] @ matrix.T, -1, 1)
        same_video = video_ids[start:stop, None] == video_ids[None, :]
        distances[same_video] = np.inf
        nearest = distances.argmin(axis=1)

        for offset, neighbor in enumerate(nearest):
            index = start + offset
            records.append({
                "filename": filenames[index],
                "video_id": video_ids[index],
                "neighbor_filename": filenames[neighbor],
                "neighbor_video_id": video_ids[neighbor],
                "distance": float(distances[offset, neighbor]),
            })

    return pd.DataFrame(records)



def cross_video_candidate_pairs(embeddings, max_theta, *, batch_size=256):
    if not np.isfinite(max_theta) or max_theta < 0:
        raise ValueError("max_theta must be finite and nonnegative")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if not embeddings:
        raise ValueError("No embeddings to analyze")

    names = np.array(sorted(embeddings))
    videos = np.array([
        Path(name).stem.rsplit("_f", 1)[0]
        for name in names
    ])

    for name in names:
        frame_number(name)

    matrix = np.stack([embeddings[name] for name in names])

    if (
        matrix.ndim != 2
        or not matrix.shape[1]
        or not np.issubdtype(matrix.dtype, np.floating)
        or not np.isfinite(matrix).all()
        or not np.allclose(
            np.linalg.norm(matrix, axis=1),
            1,
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise ValueError("Expected finite, L2-normalized embeddings")

    parts = []

    for start in range(0, len(names), batch_size):
        stop = min(start + batch_size, len(names))
        distances = 1 - np.clip(matrix[start:stop] @ matrix.T, -1, 1)

        eligible = (
            (videos[start:stop, None] != videos[None, :])
            & (
                np.arange(start, stop)[:, None]
                < np.arange(len(names))[None, :]
            )
            & (distances <= max_theta)
        )

        left, right = np.nonzero(eligible)

        parts.append(pd.DataFrame({
            "filename_a": names[start + left],
            "filename_b": names[right],
            "distance": distances[left, right],
        }))

    return pd.concat(parts, ignore_index=True).sort_values(
        ["distance", "filename_a", "filename_b"],
        ignore_index=True,
    )


def select_cross_video_frames(embeddings, pairs, theta):
    if not np.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")

    groups = {}

    for name in sorted(embeddings):
        video = Path(name).stem.rsplit("_f", 1)[0]
        groups.setdefault(video, []).append(name)

    neighbors = {}

    for names in groups.values():
        matrix = np.stack([embeddings[name] for name in names])
        distances = 1 - np.clip(matrix @ matrix.T, -1, 1)

        for index, name in enumerate(names):
            neighbors[name] = [
                (other, float(distances[index, other_index]))
                for other_index, other in enumerate(names)
                if other_index != index
            ]

    kept = set(embeddings)
    decisions = []

    ordered = pairs.loc[pairs["distance"] < theta].sort_values(
        ["distance", "filename_a", "filename_b"],
    )

    for pair in ordered.itertuples(index=False):
        a, b = pair.filename_a, pair.filename_b

        if a not in kept or b not in kept:
            continue

        video_a = Path(a).stem.rsplit("_f", 1)[0]
        video_b = Path(b).stem.rsplit("_f", 1)[0]

        if video_a == video_b:
            raise ValueError("Candidate pair belongs to the same video")

        novelty_a = min(
            (
                distance
                for name, distance in neighbors[a]
                if name in kept
            ),
            default=np.inf,
        )
        novelty_b = min(
            (
                distance
                for name, distance in neighbors[b]
                if name in kept
            ),
            default=np.inf,
        )

        if np.isinf(novelty_a) and np.isinf(novelty_b):
            continue

        removed, matched, low, high = (
            (a, b, novelty_a, novelty_b)
            if novelty_a < novelty_b
            else (b, a, novelty_b, novelty_a)
        )

        kept.remove(removed)

        decisions.append({
            "removed": removed,
            "matched_to_at_removal": matched,
            "pair_distance": pair.distance,
            "removed_novelty": low,
            "matched_novelty": high,
        })

    return sorted(kept), pd.DataFrame(
        decisions,
        columns=[
            "removed",
            "matched_to_at_removal",
            "pair_distance",
            "removed_novelty",
            "matched_novelty",
        ],
    )


def within_video_coverage(groups, selected_names):
    selected_names = set(selected_names)
    available = {
        name
        for frames in groups.values()
        for name, _ in frames
    }

    if not groups or not selected_names <= available:
        raise ValueError(
            "Selected frames must belong to the original groups"
        )

    coverage = []

    for video, frames in groups.items():
        names = [name for name, _ in frames]
        matrix = np.stack([vector for _, vector in frames])
        selected = [
            index
            for index, name in enumerate(names)
            if name in selected_names
        ]

        if not selected:
            raise ValueError(f"No retained representative for {video}")

        distances = (
            1 - np.clip(matrix @ matrix[selected].T, -1, 1)
        ).min(axis=1)
        distances[selected] = 0
        coverage.append(distances)

    return np.concatenate(coverage)


def evaluate_cross_video_candidates(
    embeddings,
    original_groups,
    theta_values,
):
    thetas = np.asarray(theta_values, dtype=float)

    if (
        thetas.ndim != 1
        or not thetas.size
        or not np.isfinite(thetas).all()
        or (thetas < 0).any()
    ):
        raise ValueError("Provide finite, nonnegative theta candidates")

    pairs = cross_video_candidate_pairs(
        embeddings,
        float(thetas.max()),
    )
    baseline_coverage = within_video_coverage(
        original_groups,
        embeddings,
    )

    records = []
    selections = {}
    decisions = {}

    for theta in np.unique(thetas):
        kept, log = select_cross_video_frames(
            embeddings,
            pairs,
            float(theta),
        )
        coverage = within_video_coverage(original_groups, kept)
        increase = np.maximum(coverage - baseline_coverage, 0)

        selections[float(theta)] = kept
        decisions[float(theta)] = log

        records.append({
            "theta": float(theta),
            "input_frames": len(embeddings),
            "kept_frames": len(kept),
            "removed_frames": len(embeddings) - len(kept),
            "removed_pct": (
                100 * (len(embeddings) - len(kept)) / len(embeddings)
            ),
            "original_coverage_p95": np.quantile(coverage, 0.95),
            "original_coverage_max": coverage.max(),
            "coverage_increase_p95": np.quantile(increase, 0.95),
            "coverage_increase_max": increase.max(),
        })

    return pd.DataFrame(records), selections, decisions