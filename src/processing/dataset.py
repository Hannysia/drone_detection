import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
from tqdm import tqdm

from src.processing.dataset_maintenance import (
    TRASH_FOLDER_NAMES,
    list_trash_folders_logic,
    restore_from_trash_logic,
    unify_labels_logic,
)
from src.processing.trackeval_dataset import (
    compile_kaggle_mot_dataset,
    prepare_trackeval_sequence,
)
from src.processing.yolo_dataset import prepare_yolo_dataset, safe_symlink
from src.utils.io import parse_video_id

log = logging.getLogger(__name__)


def _extract_metadata_fields(
    filename: str,
    *,
    normalize_background: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """Parses shape, camera, and background fields from a video or frame filename."""
    stem = parse_video_id(filename)
    if not stem:
        return None, None, None

    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "no" and parts[1] == "drone":
        shape = "no_drone"
        rest = parts[2:]
    else:
        shape = parts[0]
        rest = parts[1:]

    if not shape or shape.isdigit():
        return None, None, None

    camera = rest[0] if len(rest) >= 1 and rest[0] and not rest[0].isdigit() else None
    background = (
        rest[1]
        if camera is not None and len(rest) >= 2 and rest[1] and not rest[1].isdigit()
        else None
    )

    if normalize_background and background == "forest":
        background = "complex"

    return shape, camera, background



def rename_dataset_videos_safely(source_dir: Path, dest_dir: Path, video_extensions: list[str]) -> None:
    """Safely copies and renames classified videos into a sequential final dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    videos = [
        f for f in source_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    videos.sort()
    
    log.info(f"Files found for numbering: {len(videos)}")
    log.info("=" * 40)
    
    global_counter = 1
    for filepath in videos:
        prefix = parse_filename(filepath.name)

        if prefix is not None:
            new_filename = f"{prefix}_{global_counter:03d}{filepath.suffix}"
            new_filepath = dest_dir / new_filename
            
            try:
                shutil.copy2(filepath, new_filepath)
                log.info(f"✅ [{global_counter}/{len(videos)}] {filepath.name} -> {new_filename}")
                global_counter += 1
            except OSError as e:
                log.error(f"❌ Error copying {filepath.name}: {e}")
        else:
            log.warning(f"⏭️ Skipped (non-standard name): {filepath.name}")

def parse_filename(filename: str) -> str | None:
    """Extracts 'class_camera_background' prefix from the filename."""
    meta = extract_metadata_from_filename(filename)
    if meta is None:
        return None
    return "_".join(meta)

def perform_stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs 70/15/15 stratified split based on class and duration bins.
    Includes robust fallback logic for rare classes.
    """
    df['duration_bin'] = pd.qcut(df['frames'], q=3, labels=['short', 'med', 'long'], duplicates='drop')

    df['stratify_target'] = df['combined_class'] + "_" + df['duration_bin'].astype(str)

    strat_counts = df['stratify_target'].value_counts()
    safe_strata = strat_counts[strat_counts >= 6].index

    df['final_strat'] = df['stratify_target']
    df.loc[~df['final_strat'].isin(safe_strata), 'final_strat'] = df['combined_class']

    final_counts = df['final_strat'].value_counts()
    safe_final_classes = final_counts[final_counts >= 6].index

    df_safe = df[df['final_strat'].isin(safe_final_classes)]
    df_rare = df[~df['final_strat'].isin(safe_final_classes)]

    from sklearn.model_selection import train_test_split

    train_safe, temp_safe = train_test_split(
        df_safe, test_size=0.30, stratify=df_safe['final_strat'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_safe, test_size=0.50, stratify=temp_safe['final_strat'], random_state=42
    )
    
    train_df = pd.concat([train_safe, df_rare], ignore_index=True)
    return train_df, val_df, test_df 


def ensure_empty_split_output_dir(output_dir: Path) -> None:
    """Validates that output_dir does not exist or is an empty directory."""
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise FileExistsError(f"Split output path must be a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Split output directory is not empty: {output_dir}")


def copy_split_videos(
    source_dir: Path,
    output_dir: Path,
    splits: dict[str, pd.DataFrame],
    *,
    show_progress: bool = False,
) -> None:
    """Copies split videos atomically into an empty or new output directory."""
    ensure_empty_split_output_dir(output_dir)
    created_output = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory(dir=output_dir) as tmp:
            staged = Path(tmp)
            for split_name, split_data in splits.items():
                split_dir = staged / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                filenames = split_data["filename"].tolist()
                for filename in tqdm(
                    filenames,
                    desc=f"Copying {split_name.upper()}",
                    disable=not show_progress,
                ):
                    shutil.copy2(source_dir / filename, split_dir / filename)

            for split_name in splits:
                (staged / split_name).rename(output_dir / split_name)
    except Exception:
        if created_output and output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise



def verify_filename_syntax(
    filename: str, 
    valid_targets: list[str], 
    valid_cameras: list[str], 
    valid_bgs: list[str]
) -> str | None:
    """Verifies that the filename strictly follows the class_camera_bg pattern."""
    found_target = next((t for t in valid_targets if filename.startswith(t + "_")), None)
    if not found_target:
        return "Invalid or missing TARGET class"

    remainder1 = filename[len(found_target) + 1:] 

    found_camera = next((c for c in valid_cameras if remainder1.startswith(c + "_")), None)
    if not found_camera:
        return "Invalid CAMERA class"

    remainder2 = remainder1[len(found_camera) + 1:]

    found_bg = next((b for b in valid_bgs if remainder2.startswith(b + "_")), None)
    if not found_bg:
        return "Invalid BACKGROUND class"

    return None

def extract_metadata_from_filename(filename: str) -> tuple[str, str, str] | None:
    """Extracts (shape, camera, background) from a valid dataset filename."""
    shape, camera, background = _extract_metadata_fields(filename)
    if shape is None or camera is None or background is None:
        return None
    return shape, camera, background


def parse_frame_filename_logic(filename: str) -> dict:
    """
    Parses a flat frame filename to extract metadata (shape, camera, background).
    Handles 'no_drone' negative files safely and maps 'forest' terrain to 'complex'.
    """
    shape, camera, background = _extract_metadata_fields(
        filename, normalize_background=True
    )
    return {
        "shape": shape or "unknown",
        "camera": camera or "unknown",
        "background": background or "unknown",
    }