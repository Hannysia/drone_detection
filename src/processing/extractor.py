import os
import zipfile
import shutil
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.utils.geometry import mot_to_yolo

def parse_annotations(gt_file_path: Path) -> dict[int, list[list[float]]]:
    """Parses ground truth MOT annotations file."""
    annotations = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            f_id = int(parts[0])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            annotations.setdefault(f_id, []).append(bbox)
    return annotations

def extract_frames_and_labels(
    video_path: Path, 
    annotations: dict[int, list[list[float]]], 
    video_base_name: str, 
    pos_dir: Path, 
    neg_dir: Path
) -> None:
    """Reads video and saves positive and negative frames with YOLO labels."""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        cap.release()
        raise ValueError(f"Could not open video or dimensions are zero: {video_path}")

    frame_idx = 1
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            file_id = f"{video_base_name}_f{frame_idx:04d}"
            img_name = f"{file_id}.jpg"
            txt_name = f"{file_id}.txt"
            
            if frame_idx in annotations:
                cv2.imwrite(str(pos_dir / img_name), frame)
                label_path = pos_dir / txt_name
                with open(label_path, 'w', encoding='utf-8') as l_out:
                    for box in annotations[frame_idx]:
                        cx, cy, wn, hn = mot_to_yolo(box, w, h)
                        l_out.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
            else:
                cv2.imwrite(str(neg_dir / img_name), frame)
            
            frame_idx += 1
    finally:
        cap.release()

def download_and_extract_assets(
    repo_id: str,
    hf_token: str,
    video_path_hf: str,
    zip_path_hf: str,
    tmp_dir: Path,
    video_base_name: str,
    revision: str,
) -> tuple[Path, Path]:
    """Downloads video and zip with ground truth from Hugging Face."""
    local_video = Path(hf_hub_download(
        repo_id=repo_id, filename=video_path_hf, local_dir=tmp_dir, 
        repo_type="dataset", token=hf_token, revision=revision
    ))
    local_zip = Path(hf_hub_download(
        repo_id=repo_id, filename=zip_path_hf, local_dir=tmp_dir, 
        repo_type="dataset", token=hf_token, revision=revision
    ))

    gt_temp_path = tmp_dir / f"{video_base_name}_gt.txt"
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        internal_files = zip_ref.namelist()
        gt_internal = [n for n in internal_files if 'gt.txt' in n][0]
        with zip_ref.open(gt_internal) as src, open(gt_temp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    return local_video, gt_temp_path