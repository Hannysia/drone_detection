import gc
import logging
import os
import re
import shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download, login
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

from src.config import REPO_ID, HF_TOKEN, WORKING_DIR, VIDEO_EXTENSIONS

log = logging.getLogger(__name__)

def setup_huggingface(token: str) -> HfApi:
    """Logs in to Hugging Face and returns authenticated HfApi client."""
    login(token=token)
    return HfApi(token=token)

def get_processing_state(api: HfApi, repo_id: str, working_dir: Path) -> tuple[str | None, list[str], int, Path]:
    """Downloads processing log, scans splits on HF, and returns current split queue and next batch number."""
    files_in_repo = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    log_file_repo_path = "cvat_batches/processed_log.txt"
    log_file_local = working_dir / "processed_log.txt"
    processed_videos = set()
    
    if log_file_repo_path in files_in_repo:
        path_to_downloaded = hf_hub_download(repo_id=repo_id, filename=log_file_repo_path, repo_type="dataset")
        shutil.copy(path_to_downloaded, log_file_local)
        
        with open(log_file_local, "r", encoding="utf-8") as f:
            processed_videos = set(f.read().splitlines())
    else:
        with open(log_file_local, 'w', encoding="utf-8") as f:
            pass
        
    for current_split in ["train", "val", "test"]:
        all_split_videos = [
            f for f in files_in_repo 
            if f.startswith(f"{current_split}/") and f.endswith(tuple(VIDEO_EXTENSIONS))
        ]
        queue = [v for v in all_split_videos if v not in processed_videos]
        
        if queue:
            batch_folders = [f for f in files_in_repo if f.startswith(f"cvat_batches/{current_split}/batch_")]
            batch_nums = []
            for f in batch_folders:
                match = re.search(r'batch_(\d+)', f)
                if match: 
                    batch_nums.append(int(match.group(1)))
                    
            next_batch_num = max(batch_nums) + 1 if batch_nums else 1
            return current_split, queue, next_batch_num, log_file_local
            
    return None, [], 0, log_file_local

def create_cvat_zip(outputs_per_frame: dict, video_frames: list, video_filename: str, output_dir: Path) -> str:
    """Creates a standardized MOT format zip archive for tracking in CVAT."""
    base_dir = output_dir / "cvat_mot_tracking"
    gt_dir = base_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    with open(base_dir / "labels.txt", "w", encoding="utf-8") as f:
        f.write("drone\n")
        
    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
        for frame_idx, data in outputs_per_frame.items():
            mot_frame_idx = frame_idx + 1
            if "boxes" in data and len(data["boxes"]) > 0:
                for i, box in enumerate(data["boxes"]):
                    track_id = i + 1
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    f.write(f"{mot_frame_idx},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,1,1\n")
                    
    zip_path = output_dir / Path(video_filename).stem
    return shutil.make_archive(str(zip_path), 'zip', base_dir)

def run_sam3_labeling_pipeline(
    batch_size: int, 
    prompt_text: str, 
    ignore_list: list[str]
) -> None:
    """Runs the entire SAM3 automated bounding box labeling propagation pipeline."""
    api = setup_huggingface(HF_TOKEN)
    current_split, video_queue, batch_num, log_file = get_processing_state(api, REPO_ID, WORKING_DIR)
    
    if not current_split:
        log.info("🎉 All videos (Train, Val, Test) have been fully annotated! The task is complete.")
        return
        
    current_batch = video_queue[:batch_size]
    batch_name = f"batch_{batch_num:03d}"
    log.info(f"📁 Current split: {current_split.upper()}")
    log.info(f"📦 Creating {batch_name}. Videos in the queue for this split: {len(current_batch)}")
    
    device = Accelerator().device
    log.info("⏳ Loading SAM 3 weights...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=torch.float16).to(device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    
    local_batch_dir = WORKING_DIR / batch_name
    local_batch_dir.mkdir(parents=True, exist_ok=True)
    processed_in_this_run = []

    for remote_video_path in current_batch:
        video_filename = Path(remote_video_path).name

        if video_filename in ignore_list:
            log.warning(f"⚠️ Skipping {video_filename} (in ignore list due to OOM protection)")
            processed_in_this_run.append(remote_video_path)
            continue
        
        log.info(f"\n🎥 Processing: {video_filename}")
        
        local_vid_path = hf_hub_download(
            repo_id=REPO_ID, filename=remote_video_path, repo_type="dataset", local_dir=WORKING_DIR, token=HF_TOKEN
        )
        
        video_frames = load_video(local_vid_path)[0]
        inference_session = processor.init_video_session(
            video=video_frames, inference_device=device, processing_device="cpu", video_storage_device="cpu", dtype=torch.float16
        )
        processor.add_text_prompt(inference_session=inference_session, text=prompt_text)
        
        outputs_per_frame = {}
        for model_outputs in model.propagate_in_video_iterator(inference_session=inference_session):
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs
            
        create_cvat_zip(outputs_per_frame, video_frames, video_filename, local_batch_dir)
        processed_in_this_run.append(remote_video_path)
        Path(local_vid_path).unlink()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    log.info(f"\n🚀 Uploading {batch_name} to Hugging Face...")
    with open(log_file, "a", encoding="utf-8") as f:
        for vid in processed_in_this_run:
            f.write(f"{vid}\n")
            
    api.upload_folder(
        folder_path=str(local_batch_dir),
        path_in_repo=f"cvat_batches/{current_split}/{batch_name}",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    api.upload_file(
        path_or_fileobj=str(log_file),
        path_in_repo="cvat_batches/processed_log.txt",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    shutil.rmtree(local_batch_dir)
    log.info(f"✅ Batch {batch_name} has been successfully saved to split {current_split.upper()}.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()