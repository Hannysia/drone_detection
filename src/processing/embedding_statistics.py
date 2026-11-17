from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.processing.clustering import k_center_greedy_threshold
from src.utils.embedding_io import validate_embedding_matrix
from src.utils.io import parse_video_id


def group_embeddings_by_video(embeddings_dict: dict) -> dict[str, list]:
    """Groups frames by unique video source ID."""
    groups = defaultdict(list)
    for filename, emb in embeddings_dict.items():
        video_id = parse_video_id(filename)
        groups[video_id].append((filename, emb))
    return dict(groups)


def evaluate_theta_candidates(
    groups: dict,
    theta_values: list[float] | np.ndarray,
) -> pd.DataFrame:
    thetas = np.asarray(theta_values, dtype=float)

    if (
        thetas.ndim != 1
        or not thetas.size
        or not np.isfinite(thetas).all()
        or (thetas < 0).any()
    ):
        raise ValueError(
            "Provide a nonempty sequence of finite, nonnegative theta values",
        )

    matrices = []

    for video_id, frames in groups.items():
        if not frames:
            continue

        matrix = np.stack([
            embedding for _, embedding in frames
        ])

        validate_embedding_matrix(matrix, video_id)

        matrices.append(matrix)

    if not matrices:
        raise ValueError("No frames to evaluate")

    total = sum(len(matrix) for matrix in matrices)
    records = []

    for theta in tqdm(thetas, desc="Evaluating theta candidates"):
        kept = 0
        distances = []

        for matrix in tqdm(matrices, desc=f"theta={theta:g}", leave=False):
            selected = k_center_greedy_threshold(
                matrix,
                float(theta),
            )

            similarity = matrix @ matrix[selected].T
            nearest = (
                1.0 - np.clip(similarity, -1.0, 1.0)
            ).min(axis=1)

            nearest[selected] = 0.0
            kept += len(selected)
            distances.append(nearest)

        coverage = np.concatenate(distances)

        records.append({
            "theta": float(theta),
            "total_frames": total,
            "kept_frames": kept,
            "kept_fraction": kept / total,
            "coverage_p95": float(np.percentile(coverage, 95)),
            "coverage_max": float(coverage.max()),
        })

    return pd.DataFrame(records)