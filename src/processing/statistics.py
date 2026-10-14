import os
import logging
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from src.processing.video import get_video_metadata
from src.processing.dataset import extract_metadata_from_filename, parse_frame_filename_logic
from src.utils.visualization import plot_theta_analysis_dashboard
from pathlib import Path
from PIL import Image
import cv2
import random
from collections import defaultdict
import glob
import re 
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


log = logging.getLogger(__name__)

def analyze_video_formats(root_path):
    
    if not os.path.exists(root_path):
        print("Error: The specified path does not exist. Please check the drive letter and folder name.")
        return

    extension_counter = Counter()
    total_files = 0
    total_folders = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        total_folders += 1
        for file in filenames:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            
            if ext == "":
                ext = "no_extension"
                
            extension_counter[ext] += 1

    return extension_counter, total_folders, total_files


def get_dataset_durations_metadata(source_dir: Path, video_extensions: list[str]) -> list[dict]:
    """Gathers filename, absolute path, and duration for all valid video files."""
    durations_data = []
    
    video_files = [
        f for f in source_dir.rglob("*") 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    for video_path in video_files:
        try:
            _, _, fps, length = get_video_metadata(str(video_path))
            if fps > 0 and length > 0:
                duration = length / fps
                durations_data.append({
                    'filename': video_path.name,
                    'path': str(video_path),
                    'duration_sec': duration
                })
        except Exception:
            continue
            
    return durations_data


def get_splits_metadata_df(split_dir: Path, splits: list[str]) -> pd.DataFrame:
    """Scans all split folders, parses video filenames, and returns a single Pandas DataFrame."""
    all_data = []
    
    for split in splits:
        folder_path = split_dir / split
        if not folder_path.exists():
            continue
            
        for video_path in folder_path.iterdir():
            meta = extract_metadata_from_filename(video_path.name)
            if meta:
                shape, camera, bg = meta
                all_data.append({
                    'Class': f"{shape}_{camera}_{bg}",
                    'Morphology': shape,
                    'Camera': camera,
                    'Background': bg,
                    'Split': split.upper()
                })
                
    return pd.DataFrame(all_data)


def gather_dataset_geometry(pos_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans positive images and labels, calculates geometry dimensions and scales.
    Returns:
        - df_images: DataFrame with original widths, heights, and aspect ratios.
        - df_bboxes: DataFrame with pixel-level width, height, area, and relative scales.
    """
    img_widths = []
    img_heights = []
    img_aspect_ratios = []
    
    bbox_pixel_widths = []
    bbox_pixel_heights = []
    bbox_pixel_areas = []
    bbox_relative_areas = []
    bbox_aspect_ratios = []

    image_paths = sorted(list(pos_dir.glob("*.jpg")))
    
    for img_path in image_paths:
        lbl_path = img_path.with_suffix(".txt")

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            log.warning(f"Failed to read image header {img_path.name}: {e}")
            continue

        img_widths.append(w)
        img_heights.append(h)
        img_aspect_ratios.append(w / h)

        if lbl_path.exists():
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    _, _, _, nw, nh = map(float, parts)

                    pixel_w = nw * w
                    pixel_h = nh * h
                    pixel_area = pixel_w * pixel_h
                    relative_area = (pixel_area / (w * h)) * 100

                    bbox_pixel_widths.append(pixel_w)
                    bbox_pixel_heights.append(pixel_h)
                    bbox_pixel_areas.append(pixel_area)
                    bbox_relative_areas.append(relative_area)
                    bbox_aspect_ratios.append(pixel_w / (pixel_h + 1e-8))
            except Exception as e:
                log.warning(f"Failed to parse label {lbl_path.name}: {e}")

    df_images = pd.DataFrame({
        "width": img_widths,
        "height": img_heights,
        "aspect_ratio": img_aspect_ratios
    })

    df_bboxes = pd.DataFrame({
        "pixel_width": bbox_pixel_widths,
        "pixel_height": bbox_pixel_heights,
        "pixel_area": bbox_pixel_areas,
        "relative_area_pct": bbox_relative_areas,
        "aspect_ratio": bbox_aspect_ratios
    })

    return df_images, df_bboxes


def gather_spatial_coordinates(extracted_data_dir: Path, splits: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Scans all split positive labels and extracts normalized bounding box coordinates.
    Returns:
        - x_centers: Numpy array of normalized X coordinates clipped to [0.0, 1.0].
        - y_centers: Numpy array of normalized Y coordinates clipped to [0.0, 1.0].
        - total_boxes: Integer representing total successfully parsed bounding boxes.
    """
    x_centers = []
    y_centers = []
    total_boxes = 0

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue
            
        for path in pos_dir.glob("*.txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            x_c = float(parts[1])
                            y_c = float(parts[2])
                            
                            x_centers.append(np.clip(x_c, 0.0, 1.0))
                            y_centers.append(np.clip(y_c, 0.0, 1.0))
                            total_boxes += 1
            except Exception as e:
                log.warning(f"Failed to parse spatial labels for file {path.name}: {e}")

    return np.array(x_centers), np.array(y_centers), total_boxes


def gather_dataset_frame_metadata_df(extracted_data_dir: Path, splits: list[str]) -> pd.DataFrame:
    """
    Scans positive and negative image directories across splits,
    extracts frame-level metadata, and returns a compiled Pandas DataFrame.
    """
    data_records = []
    
    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        neg_dir = extracted_data_dir / split / "RAW_NEGATIVES"
        
        if pos_dir.exists():
            for img_path in pos_dir.glob("*.jpg"):
                meta = parse_frame_filename_logic(img_path.name)
                data_records.append({
                    "filename": img_path.name,
                    "split": split,
                    "is_positive": True,
                    "shape": meta["shape"],
                    "camera": meta["camera"],
                    "background": meta["background"],
                    "combined_class": f"{meta['shape']}_{meta['camera']}_{meta['background']}"
                })
                
        if neg_dir.exists():
            for img_path in neg_dir.glob("*.jpg"):
                meta = parse_frame_filename_logic(img_path.name)
                data_records.append({
                    "filename": img_path.name,
                    "split": split,
                    "is_positive": False,
                    "shape": "no_drone",
                    "camera": meta["camera"],
                    "background": meta["background"],
                    "combined_class": f"no_drone_{meta['camera']}_{meta['background']}"
                })
                
    return pd.DataFrame(data_records)


def gather_all_splits_geometry(extracted_data_dir: Path, splits: list[str]) -> pd.DataFrame:
    """
    Scans and gathers geometry statistics (width, height, area, aspect ratio) 
    for all bounding boxes across multiple splits. Uses fast PIL header scanner.
    """
    widths_px = []
    heights_px = []
    areas_px = []
    aspect_ratios = []

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue
            
        image_paths = sorted(list(pos_dir.glob("*.jpg")))
        for img_path in image_paths:
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, _, _, nw, nh = map(float, parts)
                            w_px = nw * w
                            h_px = nh * h
                            widths_px.append(w_px)
                            heights_px.append(h_px)
                            areas_px.append(w_px * h_px)
                            aspect_ratios.append(w_px / h_px if h_px > 0 else 1.0)
            except Exception:
                continue
                
    return pd.DataFrame({
        "pixel_width": widths_px,
        "pixel_height": heights_px,
        "pixel_area": areas_px,
        "aspect_ratio": aspect_ratios
    })


def calculate_dataset_local_contrast(extracted_data_dir: Path, splits: list[str]) -> pd.DataFrame:
    """
    Computes Local Weber Contrast for all bounding boxes in positive frames across splits.
    Extracts target luminance vs surrounding background ring luminance.
    """
    data_records = []

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue
            
        for path in pos_dir.glob("*.txt"):
            meta = parse_frame_filename_logic(path.name)
            if meta["camera"] == "unknown" or meta["background"] == "unknown":
                continue
                
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                test_path = path.with_suffix(ext)
                if test_path.exists():
                    img_path = test_path
                    break
                    
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                            x_c = float(parts[1])
                            y_c = float(parts[2])
                            
                            x1 = int((x_c - w_norm/2.0) * orig_w)
                            y1 = int((y_c - h_norm/2.0) * orig_h)
                            x2 = int((x_c + w_norm/2.0) * orig_w)
                            y2 = int((y_c + h_norm/2.0) * orig_h)
                            
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
                            
                            w_box = x2 - x1
                            h_box = y2 - y1
                            
                            if w_box <= 0 or h_box <= 0:
                                continue
                                
                            pad_w = int(w_box * 0.5)
                            pad_h = int(h_box * 0.5)
                            
                            bg_x1 = max(0, x1 - pad_w)
                            bg_y1 = max(0, y1 - pad_h)
                            bg_x2 = min(orig_w - 1, x2 + pad_w)
                            bg_y2 = min(orig_h - 1, y2 + pad_h)
                            
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            
                            target_pixels = gray[y1:y2, x1:x2]
                            mean_target = np.mean(target_pixels)
                            
                            bg_mask = np.zeros_like(gray[bg_y1:bg_y2, bg_x1:bg_x2])
                            rel_x1 = x1 - bg_x1
                            rel_y1 = y1 - bg_y1
                            rel_x2 = x2 - bg_x1
                            rel_y2 = y2 - bg_y1
                            
                            bg_mask[:, :] = 255
                            bg_mask[rel_y1:rel_y2, rel_x1:rel_x2] = 0
                            
                            bg_pixels = gray[bg_y1:bg_y2, bg_x1:bg_x2][bg_mask == 255]
                            mean_bg = np.mean(bg_pixels) if len(bg_pixels) > 0 else mean_target
                                
                            contrast = abs(mean_target - mean_bg) / max(1.0, mean_bg)
                            
                            data_records.append({
                                "camera": meta["camera"],
                                "background": meta["background"],
                                "contrast": contrast
                            })
                            
            except Exception as e:
                log.warning(f"Failed to analyze local contrast for {path.name}: {e}")

    return pd.DataFrame(data_records)



def group_embeddings_by_video(embeddings_dict: dict) -> dict[str, list]:
    """Groups frames by unique video source ID."""
    groups = defaultdict(list)
    for filename, emb in embeddings_dict.items():
        stem = Path(filename).stem  
        video_id = "_".join(stem.split("_")[:-1])  
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



def load_and_verify_embeddings(repo_id: str, filename: str, token: str | None = None) -> dict | None:
    """Downloads, loads, and performs validation metrics on a PyTorch embeddings file."""
    print(f"\n1. Downloading embeddings from Hugging Face...")
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

    print("\n2. Loading embeddings file into memory...")
    try:
        embeddings = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading PyTorch file: {e}")
        return None

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
        stem = Path(filename).stem
        video_id = re.sub(r'_f\d+$', '', stem)
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
        stem = Path(filename).stem
        video_id = re.sub(r'_f\d+$', '', stem)
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
            parts = video_id.split('_')[-3:]
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
    theta_steps: int = 50
) -> None:
    """Orchestrates the entire threshold analysis workflow for embeddings."""
    random.seed(42)
    np.random.seed(42)

    if not embeddings_dict:
        print("Error: Embeddings dictionary is empty!")
        return

    # 1. Group frames
    groups = group_embeddings_by_video(embeddings_dict)
    print(f"  {len(groups):,} unique videos found")
    print(f"  Average number of frames per video: {np.mean([len(v) for v in groups.values()]):.1f}")

    # 2. Intra-video distances (theta1)
    dist_intra = analyze_theta1_intra_distances(groups, sample_n=sample_videos)
    print(f"\nIntra-video (theta1) statistics:")
    print(f"  Min: {dist_intra.min():.4f}  Max: {dist_intra.max():.4f}  Mean: {dist_intra.mean():.4f}")

    # 3. Inter-video distances (theta2)
    dist_inter = analyze_theta2_inter_distances(embeddings_dict, sample_n=sample_pairs)
    print(f"\nInter-video (theta2) statistics:")
    print(f"  Min: {dist_inter.min():.4f}  Max: {dist_inter.max():.4f}  Mean: {dist_inter.mean():.4f}")

    # 4. Elbow curve simulation
    theta_range, kept_fractions = simulate_elbow_curve(groups, sample_n=30, theta_steps=theta_steps)

    # 5. Plot the dashboard and get optimal theta1
    elbow_theta = plot_theta_analysis_dashboard(
        dist_intra, dist_inter, theta_range, kept_fractions
    )

    print(f"\n{'='*50}\nRECOMMENDATIONS FOR THETA\n{'='*50}")
    
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
    print(f"                after running Greedy selection, visually verify results on 2-3 videos")