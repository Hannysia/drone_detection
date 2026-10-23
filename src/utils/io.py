import glob
import os
import shutil
import zipfile
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def get_processed_list(log_file: Path) -> set[str]:
    """Reads the log file to get names of already processed videos."""
    if not log_file.exists():
        return set()
    with open(log_file, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def mark_as_processed(log_file: Path, video_name: str) -> None:
    """Writes the successfully processed video name to the log."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(video_name + '\n')

def cleanup_partial_files(video_name: str, pos_dir: Path, neg_dir: Path) -> None:
    """Removes frames and labels if the previous process was interrupted."""
    patterns = [
        pos_dir / f"{video_name}_f*.jpg",
        pos_dir / f"{video_name}_f*.txt",
        neg_dir / f"{video_name}_f*.jpg"
    ]
    for pattern in patterns:
        for f in glob.glob(str(pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def read_yolo_bbox(label_path: Path) -> list[float]:
    """Reads first bbox from YOLO label file. Returns [cx, cy, w, h]."""
    with open(label_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Label file is empty: {label_path}")
        parts = first_line.split()
    return list(map(float, parts[1:5]))

def move_to_trash(file_path: Path, trash_dir: Path, is_positive: bool = True) -> None:
    """Moves image (.jpg) and optionally its YOLO label (.txt) to trash directory."""
    try:
        if is_positive:
            label_path = file_path.with_suffix('.txt')
            if label_path.exists():
                shutil.move(str(label_path), trash_dir / label_path.name)
        
        img_path = file_path.with_suffix('.jpg')
        if img_path.exists():
            shutil.move(str(img_path), trash_dir / img_path.name)
            
    except OSError as e:
        raise OSError(f"Error moving file {file_path.name} to trash: {e}")



def create_zip_archive(
    split_name: str, 
    source_dir: Path, 
    output_zip_path: Path, 
    subdirs: list[str]
) -> bool:
    """Archives specific format files (.jpg, .txt) from subdirs into a single ZIP file."""
    log.info(f"[{split_name.upper()}] Archiving files...")
    
    file_count = 0
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for subdir in subdirs:
            target_path = source_dir / subdir
            if not target_path.exists():
                log.warning(f"Directory {target_path} does not exist, skipping.")
                continue
                
            for file_path in target_path.glob('*'):
                if file_path.suffix in ['.jpg', '.txt']:
                    arcname = Path(subdir) / file_path.name
                    zipf.write(file_path, arcname)
                    file_count += 1
                    
    log.info(f"[{split_name.upper()}] Archive created: {output_zip_path} ({file_count} files packed)")
    return file_count > 0



def get_extraction_paths(
    baseline_dir: Path,
    historical_dir: Path,
) -> tuple[Path, Path, Path]:
    """Return isolated extraction paths without modifying files."""

    baseline = baseline_dir.expanduser().resolve()
    historical = historical_dir.expanduser().resolve()

    if (
        baseline == historical
        or baseline.is_relative_to(historical)
        or historical.is_relative_to(baseline)
    ):
        raise ValueError(
            "Extraction workspace must not overlap "
            "the historical dataset: "
            f"{baseline} vs {historical}"
        )

    output_dir = baseline
    tmp_dir = baseline / ".tmp"
    log_file = baseline / ".state" / "processed_videos.log"

    return output_dir, tmp_dir, log_file
