import os
import cv2
import zipfile
import numpy as np
from pathlib import Path

# Input directories
BASE_DIR = Path("E:/bachelor_data/labels")

# Output directory in E:/bachelor_data
OUTPUT_DIR = Path("E:/bachelor_data/drone_mot_dataset")
SEQMAPS_DIR = OUTPUT_DIR / "seqmaps"
SEQMAPS_DIR.mkdir(parents=True, exist_ok=True)

# Splits to process
SPLITS = ["test", "val"]
video_extensions = ('.mp4', '.mov', '.m4v')

# Find all zip files in BASE_DIR
zip_files = {z.stem: z for z in BASE_DIR.glob("*.zip")}

for split in SPLITS:
    split_dir = BASE_DIR / split
    if not split_dir.exists():
        print(f"Warning: Split directory {split_dir} does not exist. Skipping.")
        continue
        
    print(f"\n--- Processing split: {split} ---")
    
    # TrackEval naming standard: <BENCHMARK>-<SPLIT>
    trackeval_split_name = f"drone-{split}"
    gt_dir = OUTPUT_DIR / trackeval_split_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    videos = [f for f in split_dir.iterdir() if f.suffix.lower() in video_extensions]
    matched_seqs = []
    
    for video_path in videos:
        seq_name = video_path.stem
        # Check for exact name match or with "_annotated" suffix
        zip_path = zip_files.get(seq_name) or zip_files.get(f"{seq_name}_annotated")
        
        if zip_path:
            matched_seqs.append(seq_name)
            print(f"[{split.upper()}] Processing sequence: {seq_name}")
            
            # 1. Get video metadata for seqinfo.ini
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 2. Create target sequence folder
            seq_gt_dir = gt_dir / seq_name / "gt"
            seq_gt_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. Write seqinfo.ini
            seqinfo_content = f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate={int(fps)}\nseqLength={length}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n"
            with open(gt_dir / seq_name / "seqinfo.ini", "w") as f:
                f.write(seqinfo_content)
                
            # 4. Extract and normalize annotations directly from ZIP stream
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                gt_file_in_zip = None
                for name in zip_ref.namelist():
                    if name.endswith("gt.txt"):
                        gt_file_in_zip = name
                        break
                
                if gt_file_in_zip:
                    # Read directly into numpy array
                    with zip_ref.open(gt_file_in_zip) as f_stream:
                        gt_data = np.loadtxt(f_stream, delimiter=',')
                    
                    # Enforce class ID to 1 (TrackEval standardization)
                    if gt_data.ndim == 2:
                        gt_data[:, 7] = 1
                    elif gt_data.ndim == 1:
                        gt_data[7] = 1
                    
                    # Save processed data
                    target_gt_path = seq_gt_dir / "gt.txt"
                    np.savetxt(target_gt_path, gt_data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f')
                else:
                    print(f"Warning: gt.txt was not found inside {zip_path.name}")
        else:
            print(f"Warning: No matching zip archive found for {video_path.name}")
            
    # Write seqmaps file for this specific split
    with open(SEQMAPS_DIR / f"{trackeval_split_name}.txt", "w") as f:
        f.write("name\n" + "\n".join(matched_seqs))

print(f"\nDataset generation complete. Saved to: {OUTPUT_DIR}")