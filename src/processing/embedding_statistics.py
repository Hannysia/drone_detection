from collections import defaultdict
import glob
import logging
from pathlib import Path
import random

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.processing.clustering import k_center_greedy_threshold
from src.utils.embedding_io import load_embedding_file, validate_embedding_matrix
from src.utils.io import parse_video_id
from src.utils.visualization import plot_theta_analysis_dashboard

log = logging.getLogger(__name__)


def group_embeddings_by_video(embeddings_dict: dict) -> dict[str, list]:
    """Groups frames by unique video source ID."""
    groups = defaultdict(list)
    for filename, emb in embeddings_dict.items():
        video_id = parse_video_id(filename)
        groups[video_id].append((filename, emb))
    return dict(groups)


def pairwise_distances(embeddings_array: np.ndarray) -> np.ndarray:
    """Calculates pairwise cosine distances for a set of embeddings."""
    sim = embeddings_array @ embeddings_array.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    idx = np.triu_indices(len(embeddings_array), k=1)
    return dist[idx]


def analyze_theta1_intra_distances(groups: dict, sample_n: int) -> np.ndarray:
    """Extracts pairwise distances within the same video files."""
    video_ids = random.sample(list(groups.keys()), min(sample_n, len(groups)))
    all_distances = []

    for vid in video_ids:
        frames = groups[vid]
        if len(frames) < 2:
            continue
        embs = np.stack([e for _, e in frames])
        dists = pairwise_distances(embs)
        all_distances.append(dists)

    return np.concatenate(all_distances)


def analyze_theta2_inter_distances(embeddings_dict: dict, sample_n: int) -> np.ndarray:
    """Extracts pairwise distances between completely different video sequences."""
    all_embs = np.stack(list(embeddings_dict.values()))
    idx1 = np.random.randint(0, len(all_embs), sample_n)
    idx2 = np.random.randint(0, len(all_embs), sample_n)

    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    return 1.0 - np.clip(
        np.sum(all_embs[idx1] * all_embs[idx2], axis=1), -1.0, 1.0
    )


def simulate_elbow_curve(groups: dict, sample_n: int, theta_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulates frames retention rate across a wide range of theta1 thresholds."""
    video_ids = random.sample(list(groups.keys()), min(sample_n, len(groups)))
    theta_range = np.linspace(0.01, 0.8, theta_steps)
    kept_fractions = []

    for theta in theta_range:
        total, kept = 0, 0
        for vid in video_ids:
            frames = groups[vid]
            if len(frames) < 2:
                kept += len(frames)
                total += len(frames)
                continue
            embs = np.stack([e for _, e in frames])
            sim = embs @ embs.T
            np.fill_diagonal(sim, -2.0)
            nearest = 1.0 - sim.max(axis=1)
            kept += int((nearest >= theta).sum())
            total += len(frames)

        kept_fractions.append(kept / total if total > 0 else 0)

    return theta_range, np.array(kept_fractions)


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


def load_and_verify_embeddings(repo_id: str, filename: str, token: str | None = None) -> dict:
    """Downloads, loads, and performs validation metrics on a PyTorch embeddings file."""
    log.info("1. Downloading embeddings from Hugging Face...")
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    except Exception as exc:
        log.error("Failed to download embeddings %s from %s: %s", filename, repo_id, exc)
        raise

    log.info("2. Loading embeddings file into memory...")
    try:
        embeddings = load_embedding_file(Path(file_path))
    except Exception as exc:
        log.error("Failed to load embeddings file %s: %s", file_path, exc)
        raise

    if not embeddings:
        raise ValueError(f"Embeddings dictionary is empty: {filename}")

    total_frames = len(embeddings)
    print(f"\n3. Analysis:")
    print(f"  - Total frames parsed: {total_frames}")

    first_key = list(embeddings.keys())[0]
    first_vector = embeddings[first_key]

    print(f"  - Key format example: '{first_key}'")
    print(f"  - Vector shape: {first_vector.shape}")

    first_vector_np = first_vector.numpy() if isinstance(first_vector, torch.Tensor) else np.array(first_vector)
    l2_norm = np.linalg.norm(first_vector_np)
    print(f"  - L2 Norm of vector: {l2_norm:.6f} (Should be ~1.0 if normalized)")

    has_nan = np.isnan(first_vector_np).any()
    has_inf = np.isinf(first_vector_np).any()
    print(f"  - Contains NaN: {has_nan}")
    print(f"  - Contains Inf: {has_inf}")

    return embeddings


def collect_surviving_frames(data_dir: Path) -> set[str]:
    """Collects filenames of images that survived the filtering stage."""
    pattern = str(data_dir / "*.jpg")
    files = glob.glob(pattern)
    names = {Path(f).name for f in files}
    print(f"Found {len(names):,} frames")
    return names


def filter_active_embeddings(embeddings_dict: dict, surviving_names: set[str]) -> dict[str, np.ndarray]:
    """Filters the in-memory embeddings dictionary to retain only surviving active frames."""
    filtered = {}
    for k, v in embeddings_dict.items():
        if k in surviving_names:
            filtered[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
    print(f"Filtered embeddings: {len(filtered):,} active vectors (down from {len(embeddings_dict):,})")
    return filtered


def group_filtered_by_video_regex(embeddings_dict: dict) -> dict[str, list]:
    """Groups filtered embeddings by their parsed regex video ID."""
    groups = defaultdict(list)
    for filename, emb in embeddings_dict.items():
        video_id = parse_video_id(filename)
        groups[video_id].append(emb)

    print(f"Grouped into {len(groups):,} unique videos")
    sizes = [len(v) for v in groups.values()]
    print(f"Frames per video — Mean: {np.mean(sizes):.1f}, Min: {np.min(sizes)}, Max: {np.max(sizes)}")
    return dict(groups)


def compute_inter_video_distances(groups_dict: dict, sample_n: int = 50_000) -> np.ndarray:
    """Computes cross-video pairwise cosine distances between random video sequences."""
    print(f"\nComputing {sample_n:,} inter-video pairwise distances...")

    video_ids = list(groups_dict.keys())
    all_embs_per_video = [np.stack(groups_dict[v]) for v in video_ids]

    dists = []
    rng = np.random.default_rng(42)

    with tqdm(total=sample_n, desc="Sampling inter-video pairs") as pbar:
        while len(dists) < sample_n:
            i, j = rng.choice(len(video_ids), size=2, replace=False)
            embs_i = all_embs_per_video[i]
            embs_j = all_embs_per_video[j]

            ei = embs_i[rng.integers(len(embs_i))]
            ej = embs_j[rng.integers(len(embs_j))]

            dist = 1.0 - float(np.clip(ei @ ej, -1.0, 1.0))
            dists.append(dist)
            pbar.update(1)

    dists = np.array(dists)
    print(f"Min: {dists.min():.4f}  Max: {dists.max():.4f}  Mean: {dists.mean():.4f}  Median: {np.median(dists):.4f}")
    return dists


def diagnose_embeddings_contents(embeddings_dict: dict, suspect_videos: list[str]) -> None:
    """Performs deep diagnostics on embeddings keys format and searches for suspected missing videos."""
    print(f"Total keys found in dictionary: {len(embeddings_dict):,}\n")

    video_groups = defaultdict(list)
    for filename in embeddings_dict.keys():
        video_id = parse_video_id(filename)
        video_groups[video_id].append(filename)

    print(f"Unique videos found in dictionary: {len(video_groups):,}")
    print()

    print("=" * 60)
    print("SUSPECT VIDEOS CHECK")
    print("=" * 60)

    for video_id in suspect_videos:
        if video_id in video_groups:
            frames = video_groups[video_id]
            print(f"✓ FOUND: {video_id}")
            print(f"  Total frames parsed: {len(frames)}")
            print(f"  Example key: '{frames[0]}'")
        else:
            print(f"✗ MISSING: {video_id}")
            parts = video_id.split("_")[-3:]
            similar = [
                v for v in video_groups.keys()
                if any(part in v for part in parts)
            ]
            if similar:
                print(f"  Similar names found: {similar[:5]}")
        print()

    print("=" * 60)
    print("ALL UNIQUE SEQUENCES IN EMBS DICTIONARY")
    print("=" * 60)
    for video_id, frames in sorted(video_groups.items()):
        print(f"  {video_id}: {len(frames)} frames")

    print()
    print("=" * 60)
    print("KEY SYNTAX TEMPLATE SAMPLES (First 10)")
    print("=" * 60)
    for key in list(embeddings_dict.keys())[:10]:
        print(f"  '{key}'")


def run_theta_analysis(
    embeddings_dict: dict,
    sample_videos: int = 50,
    sample_pairs: int = 50000,
    theta_steps: int = 50,
) -> None:
    """Orchestrates the entire threshold analysis workflow for embeddings."""
    random.seed(42)
    np.random.seed(42)

    if not embeddings_dict:
        print("Error: Embeddings dictionary is empty!")
        return

    groups = group_embeddings_by_video(embeddings_dict)
    print(f"  {len(groups):,} unique videos found")
    print(f"  Average number of frames per video: {np.mean([len(v) for v in groups.values()]):.1f}")

    dist_intra = analyze_theta1_intra_distances(groups, sample_n=sample_videos)
    print(f"\nIntra-video (theta1) statistics:")
    print(f"  Min: {dist_intra.min():.4f}  Max: {dist_intra.max():.4f}  Mean: {dist_intra.mean():.4f}")

    dist_inter = analyze_theta2_inter_distances(embeddings_dict, sample_n=sample_pairs)
    print(f"\nInter-video (theta2) statistics:")
    print(f"  Min: {dist_inter.min():.4f}  Max: {dist_inter.max():.4f}  Mean: {dist_inter.mean():.4f}")

    theta_range, kept_fractions = simulate_elbow_curve(groups, sample_n=30, theta_steps=theta_steps)

    elbow_theta = plot_theta_analysis_dashboard(
        dist_intra, dist_inter, theta_range, kept_fractions
    )

    print(f"\n{'=' * 50}\nRECOMMENDATIONS FOR THETA\n{'=' * 50}")

    theta1_conservative = np.percentile(dist_intra, 10)
    theta1_moderate = np.percentile(dist_intra, 25)
    print(f"theta1 (video-level threshold):")
    print(f"  Conservative (p10):   {theta1_conservative:.4f}  — removes ~10% of the most static pairs")
    print(f"  Elbow curve:          {elbow_theta:.4f}  — optimal balance of dataset size & diversity")
    print(f"  Moderate     (p25):   {theta1_moderate:.4f}  — more aggressive frame removal")

    theta2_conservative = np.percentile(dist_inter, 15)
    theta2_moderate = np.percentile(dist_inter, 30)
    print(f"\ntheta2 (dataset-level threshold):")
    print(f"  Conservative (p15):   {theta2_conservative:.4f}  — removes redundant duplicate scenes")
    print(f"  Moderate     (p30):   {theta2_moderate:.4f}  — more aggressive dataset pruning")

    print(f"\nRecommendation: start with theta1={elbow_theta:.3f} and theta2={theta2_conservative:.3f}")
    print("                after running Greedy selection, visually verify results on 2-3 videos")