import os
import cv2
import zipfile
import shutil
import glob
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIGURATION =================
REPO_ID = os.getenv("REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN") 
LOCAL_BASE_DIR = Path(os.getenv("LOCAL_BASE_DIR"))
OUTPUT_DIR = os.path.join(LOCAL_BASE_DIR, "extracted_dataset")
LOG_FILE = os.path.join(LOCAL_BASE_DIR, "processed_videos.log")
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v']
from src.config import SPLITS

api = HfApi(token=HF_TOKEN)

def mot_to_yolo(box, img_width, img_height):
    """Converts MOT bounding box to YOLO normalized format."""
    x_left, y_top, w, h = box
    cx = (x_left + w / 2) / img_width
    cy = (y_top + h / 2) / img_height
    wn = w / img_width
    hn = h / img_height
    return cx, cy, wn, hn

def get_processed_list():
    """Reads the log file to get names of already processed videos."""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f)

def mark_as_processed(video_name):
    """Writes the successfully processed video name to the log."""
    with open(LOG_FILE, 'a') as f:
        f.write(video_name + '\n')

def cleanup_partial_files(video_name, pos_dir, neg_dir):
    """Removes frames and labels if the previous process was interrupted."""
    patterns = [
        os.path.join(pos_dir, f"{video_name}_f*.jpg"),
        os.path.join(pos_dir, f"{video_name}_f*.txt"),
        os.path.join(neg_dir, f"{video_name}_f*.jpg")
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

print("Fetching file list from Hugging Face repository...")
repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
processed_videos = get_processed_list()

for split in SPLITS:
    print(f"\n" + "="*80)
    print(f"STARTING SPLIT: {split.upper()}")
    print("="*80)

    hf_labels_root = f"labels/{split}/"
    split_pos_dir = os.path.join(OUTPUT_DIR, split, "RAW_POSITIVES")
    split_neg_dir = os.path.join(OUTPUT_DIR, split, "RAW_NEGATIVES")
    tmp_dir = os.path.join(LOCAL_BASE_DIR, "tmp", split)

    os.makedirs(split_pos_dir, exist_ok=True)
    os.makedirs(split_neg_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    zip_tasks = [f for f in repo_files if f.startswith(hf_labels_root) and f.endswith('.zip')]

    for zip_path in zip_tasks:
        video_base_name = os.path.basename(zip_path).replace('.zip', '')
        
        if video_base_name in processed_videos:
            print(f"Skipping {video_base_name}: Already DONE.")
            continue

        video_path_hf = None
        for ext in VIDEO_EXTENSIONS:
            potential_path = f"{split}/{video_base_name}{ext}"
            match = [f for f in repo_files if f.lower() == potential_path.lower()]
            if match:
                video_path_hf = match[0]
                break

        if not video_path_hf:
            print(f"❌ NOT FOUND: Video for {video_base_name} in {split}/ with extensions {VIDEO_EXTENSIONS}")
            continue

        cleanup_partial_files(video_base_name, split_pos_dir, split_neg_dir)

        print(f"\nProcessing: {video_base_name} (Format: {os.path.splitext(video_path_hf)[1]})")
        
        try:
            local_video = hf_hub_download(
                repo_id=REPO_ID, filename=video_path_hf, local_dir=tmp_dir, 
                repo_type="dataset", token=HF_TOKEN
            )
            local_zip = hf_hub_download(
                repo_id=REPO_ID, filename=zip_path, local_dir=tmp_dir, 
                repo_type="dataset", token=HF_TOKEN
            )

            gt_temp_path = os.path.join(tmp_dir, f"{video_base_name}_gt.txt")
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                internal_files = zip_ref.namelist()
                gt_internal = [n for n in internal_files if 'gt.txt' in n][0]
                with zip_ref.open(gt_internal) as src, open(gt_temp_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

            annotations = {}
            with open(gt_temp_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    f_id = int(parts[0])
                    bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                    if f_id not in annotations:
                        annotations[f_id] = []
                    annotations[f_id].append(bbox)

            cap = cv2.VideoCapture(local_video)
            frame_idx = 1
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if w == 0 or h == 0:
                print(f"⚠️ Error: Could not open video {video_base_name}")
                cap.release()
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                file_id = f"{video_base_name}_f{frame_idx:04d}"
                img_name = f"{file_id}.jpg"
                txt_name = f"{file_id}.txt"
                
                if frame_idx in annotations:
                    cv2.imwrite(os.path.join(split_pos_dir, img_name), frame)
                    
                    label_path = os.path.join(split_pos_dir, txt_name)
                    with open(label_path, 'w') as l_out:
                        for box in annotations[frame_idx]:
                            cx, cy, wn, hn = mot_to_yolo(box, w, h)
                            l_out.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
                else:
                    cv2.imwrite(os.path.join(split_neg_dir, img_name), frame)
                
                frame_idx += 1
            
            cap.release()
            
            mark_as_processed(video_base_name)
            
            os.remove(local_video)
            os.remove(local_zip)
            os.remove(gt_temp_path)
            print(f"✅ Success: {video_base_name}")

        except Exception as e:
            print(f"⚠️ ERROR processing {video_base_name}: {e}")

print("\n" + "="*80)
print("PROCESS FINISHED")
print(f"Check {LOG_FILE} for the list of completed videos.")