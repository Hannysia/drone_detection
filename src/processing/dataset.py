import logging
import shutil
from pathlib import Path
import pandas as pd
import zipfile
import numpy as np
from src.processing.video import get_video_metadata

log = logging.getLogger(__name__)

TRASH_FOLDER_NAMES = {
    "kcenter": "local_kcenter_{type}",
    "uniform": "uniform_sample_{type}",
    "iou":     "iou_dedup_{type}",
}

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
        old_name = filepath.stem
        parts = old_name.split('_')
        
        if len(parts) >= 3:
            prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
            
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
    name_no_ext = filename.split('.')[0]
    parts = name_no_ext.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None

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


def prepare_trackeval_sequence(
    video_path: Path, 
    zip_path: Path, 
    seq_gt_dir: Path, 
    seqinfo_path: Path
) -> None:
    """Extracts ground truth from zip, standardizes category ID to 1, and writes metadata."""
    width, height, fps, length = get_video_metadata(str(video_path))
    
    seq_name = video_path.stem
    seqinfo_content = (
        f"[Sequence]\nname={seq_name}\nimDir=img1\n"
        f"frameRate={int(fps)}\nseqLength={length}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n"
    )
    with open(seqinfo_path, "w", encoding="utf-8") as f:
        f.write(seqinfo_content)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        gt_file_in_zip = next((name for name in zip_ref.namelist() if name.endswith("gt.txt")), None)
        
        if gt_file_in_zip:
            with zip_ref.open(gt_file_in_zip) as f_stream:
                gt_data = np.loadtxt(f_stream, delimiter=',')
            
            if gt_data.ndim == 2:
                gt_data[:, 7] = 1
            elif gt_data.ndim == 1:
                gt_data[7] = 1
            
            target_gt_path = seq_gt_dir / "gt.txt"
            np.savetxt(target_gt_path, gt_data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f')
        else:
            log.warning(f"Warning: gt.txt was not found inside {zip_path.name}")


def list_trash_folders_logic(trash_dir: Path) -> None:
    """Lists all available trash folders and file counts."""
    if not trash_dir.exists():
        log.error(f"TRASH_DIR does not exist: {trash_dir}")
        return

    log.info(f"\nAll trash folders in {trash_dir}:")
    log.info("=" * 60)

    found = False
    for split_dir in sorted(trash_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for trash_subdir in sorted(split_dir.iterdir()):
            if not trash_subdir.is_dir():
                continue
            jpg_count = len(list(trash_subdir.glob("*.jpg")))
            txt_count = len(list(trash_subdir.glob("*.txt")))
            log.info(f"  {split_dir.name}/{trash_subdir.name}/")
            log.info(f"    .jpg: {jpg_count}  .txt: {txt_count}")
            found = True

    if not found:
        log.info("  (empty)")

def restore_from_trash_logic(
    split: str, 
    embedding_type: str, 
    stage: str, 
    trash_base_dir: Path, 
    extracted_data_dir: Path,
    dry_run: bool = False
) -> None:
    """Moves back deleted files from a specific trash stage to the working directory."""
    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"

    folder_template = TRASH_FOLDER_NAMES.get(stage)
    if folder_template is None:
        log.error(f"Unknown stage: {stage}. Available: {list(TRASH_FOLDER_NAMES.keys())}")
        return

    trash_folder_name = folder_template.replace("{type}", embedding_type)
    trash_dir = trash_base_dir / split / trash_folder_name
    target_data_dir = extracted_data_dir / split / subdir_name

    if not trash_dir.exists():
        log.error(f"Trash directory not found: {trash_dir}")
        return

    if not target_data_dir.exists():
        log.error(f"Target directory not found: {target_data_dir}")
        return

    trash_files = sorted(list(trash_dir.glob("*.jpg")))
    total = len(trash_files)

    if total == 0:
        log.info(f"Trash is empty: {trash_dir}")
        return

    log.info(f"\nRestore: {split.upper()} / {embedding_type.upper()} / stage={stage}")
    log.info(f"  Source:      {trash_dir}")
    log.info(f"  Destination: {target_data_dir}")
    log.info(f"  Files .jpg:  {total}")

    if dry_run:
        log.info("\n[DRY RUN] Changes are not being applied. First 10 files:")
        for f in trash_files[:10]:
            log.info(f"  {f.name} → {target_data_dir / f.name}")
        if total > 10:
            log.info(f"  ... and {total - 10} more files")
        return

    restored_jpg = 0
    restored_txt = 0
    errors = 0

    from tqdm.auto import tqdm
    for jpg_path in tqdm(trash_files, desc=f"Restoring {split}/{embedding_type}"):
        try:
            dest_jpg = target_data_dir / jpg_path.name
            shutil.move(str(jpg_path), dest_jpg)
            restored_jpg += 1

            if is_positive:
                txt_path = trash_dir / jpg_path.with_suffix(".txt").name
                if txt_path.exists():
                    shutil.move(str(txt_path), target_data_dir / txt_path.name)
                    restored_txt += 1

        except OSError as e:
            log.error(f"Error: {jpg_path.name}: {e}")
            errors += 1

    log.info(f"\n[{split.upper()}] Restore completed.")
    log.info(f"  Restored .jpg: {restored_jpg}/{total}")
    if is_positive:
        log.info(f"  Restored .txt: {restored_txt}")
    if errors:
        log.warning(f"  Errors: {errors}")


def unify_labels_logic(base_path: Path, splits: list[str], force_class_id: str) -> None:
    """Goes through all label files and forces the category class ID to force_class_id."""
    log.info(f"Starting label unification at: {base_path}")
    total_updated = 0

    for split in splits:
        label_dir = base_path / split / "RAW_POSITIVES"
        
        if not label_dir.exists():
            log.warning(f"Warning: Directory {label_dir} not found. Skipping {split}...")
            continue

        log.info(f"Processing split: {split}...")
        
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue

                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        parts[0] = force_class_id
                        updated_lines.append(" ".join(parts) + "\n")
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)
                
                total_updated += 1
            except Exception as e:
                log.error(f"Failed to overwrite label {label_file.name}: {e}")

    log.info("\n" + "="*50)
    log.info("UNIFICATION COMPLETE")
    log.info(f"Total files updated to Class ID {force_class_id}: {total_updated}")
    log.info("="*50)



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
    name_without_ext = filename.split('.')[0]
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2 and parts[0] == 'no' and parts[1] == 'drone':
        shape = 'no_drone'
        if len(parts) >= 4:
            camera = parts[2]
            bg = parts[3]
        else:
            return None
    else:
        if len(parts) >= 4:
            shape = parts[0]
            camera = parts[1]
            bg = parts[2]
        else:
            return None
            
    return shape, camera, bg


def parse_frame_filename_logic(filename: str) -> dict:
    """
    Parses a flat frame filename to extract metadata (shape, camera, background).
    Handles 'no_drone' negative files safely and maps 'forest' terrain to 'complex'.
    """
    clean_name = filename.replace("no_drone", "nodrone")
    parts = clean_name.split('_')
    
    if parts[0] == "nodrone":
        shape = "no_drone"
        camera = parts[1] if len(parts) > 1 else "unknown"
        if len(parts) > 2:
            background = "unknown" if parts[2].isdigit() else parts[2]
        else:
            background = "unknown"
    else:
        shape = parts[0]
        camera = parts[1] if len(parts) > 1 else "unknown"
        background = parts[2] if len(parts) > 2 else "unknown"
        
    if background == "forest":
        background = "complex"
        
    return {
        "shape": shape,
        "camera": camera,
        "background": background
    }



def compile_kaggle_mot_dataset(local_hf_dir: Path, mot_dataset_dir: Path, seqmaps_dir: Path, video_extensions: tuple[str]) -> None:
    """Compiles local Kaggle directories and extracts MOT labels to build TrackEval dataset."""
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    splits = ["test", "val"]

    labels_root = local_hf_dir / "labels"
    zip_files = {z.stem: z for z in labels_root.rglob("*.zip")}
    print(f"Found {len(zip_files)} label archives in recursive search.")

    for split in splits:
        split_video_dir = local_hf_dir / split
        trackeval_split_name = f"drone-{split}"
        gt_dir = mot_dataset_dir / trackeval_split_name
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        videos = [f for f in split_video_dir.iterdir() if f.suffix.lower() in video_extensions]
        matched_seqs = []
        
        for video_path in videos:
            seq_name = video_path.stem
            zip_path = zip_files.get(seq_name) or zip_files.get(f"{seq_name}_annotated")
            
            if zip_path:
                matched_seqs.append(seq_name)
                seq_gt_dir = gt_dir / seq_name / "gt"
                seq_gt_dir.mkdir(parents=True, exist_ok=True)
                seqinfo_path = gt_dir / seq_name / "seqinfo.ini"
                
                prepare_trackeval_sequence(video_path, zip_path, seq_gt_dir, seqinfo_path)
            else:
                print(f"Warning: No matching zip archive found for video {video_path.name}")
                
        with open(seqmaps_dir / f"{trackeval_split_name}.txt", "w", encoding="utf-8") as f:
            f.write("name\n" + "\n".join(matched_seqs))
            
    print("Dataset compilation for TrackEval complete.")