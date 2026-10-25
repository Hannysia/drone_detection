import logging
import re
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_download

from src.config import (
    REPO_ID, CVAT_HOST, CVAT_PORT, LABELS_DIR, CVAT_USER, CVAT_PASS, HF_TOKEN, VIDEO_EXTENSIONS
)
from src.utils.shell import run_cmd

log = logging.getLogger(__name__)
api = HfApi(token=HF_TOKEN)

ANNOTATION_SOURCES = {
    "sam3": ("cvat_batches", "_mot.zip"),
    "labels": ("labels", ".zip"),
}

def pull_batch_to_cvat(split: str, batch_name: str, annotation_source: str = "sam3") -> None:
    """Downloads a batch from HF and automatically creates CVAT tasks for all videos."""
    log.info(f"📥 Searching for {batch_name} in split '{split}' on Hugging Face...")
    
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(
            f"Unknown annotation source: {annotation_source}"
        )

    source_dir, suffix = ANNOTATION_SOURCES[annotation_source]
    batch_prefix = f"{source_dir}/{split}/{batch_name}/"

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    zip_files = [
        path
        for path in files
        if path.startswith(batch_prefix)
        and Path(path).parent.as_posix() == batch_prefix.rstrip("/")
        and path.endswith(suffix)
    ]
    
    if not zip_files:
        log.error("❌ No annotation archives found in this batch.")
        return

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for zip_file_repo_path in zip_files:
        zip_filename = Path(zip_file_repo_path).name
        base_name = zip_filename[:-len(suffix)]
        
        video_repo_path = next(
            (f for f in files if f.startswith(f"{split}/{base_name}.") and f.endswith(tuple(VIDEO_EXTENSIONS))), 
            None
        )
        
        if not video_repo_path:
            log.warning(f"⚠️ Skipping {base_name}: could not find the original video on HF.")
            continue
            
        log.info(f"\n⏳ Downloading {base_name}...")
        local_video_path = hf_hub_download(
            repo_id=REPO_ID, filename=video_repo_path, repo_type="dataset", local_dir=LABELS_DIR, token=HF_TOKEN
        )
        local_zip_path = hf_hub_download(
            repo_id=REPO_ID, filename=zip_file_repo_path, repo_type="dataset", local_dir=LABELS_DIR, token=HF_TOKEN
        )

        log.info(f"🚀 Creating a CVAT task for {base_name}...")
        task_name = f"{base_name}_{split}"
        
        create_cmd = (
            f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
            f'--auth {CVAT_USER}:{CVAT_PASS} create "{task_name}" '
            f'--labels "[{{\\"name\\": \\"pedestrian\\\"}}, {{\\"name\\": \\"drone\\\"}}]" '
            f'local "{local_video_path}"'
        )
        output = run_cmd(create_cmd)
        
        match = re.search(r'(?i)ID[\s:]*(\d+)', output)
        if not match:
            log.error(f"❌ Failed to extract Task ID for {base_name}.")
            log.error(f"🔍 Raw CVAT Output:\n{output.strip()}")
            continue
            
        task_id = match.group(1)
        log.info(f"✅ Task created! Task ID: {task_id}")

        log.info(f"📦 Uploading {annotation_source} annotations to task...")
        upload_cmd = (
            f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
            f'--auth {CVAT_USER}:{CVAT_PASS} upload {task_id} "{local_zip_path}" --format "MOT 1.1"'
        )
        run_cmd(upload_cmd)
        
        log.info(f"🎉 Done: http://{CVAT_HOST}:{CVAT_PORT}/tasks/{task_id}")

def push_batch_to_hf(task_id: str, split: str, batch_name: str) -> None:
    """Exports validated annotations from CVAT and pushes them to HF labels."""
    log.info(f"📤 Exporting annotation from CVAT (Task ID: {task_id})...")
    
    try:
        response = requests.get(
            f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
            auth=(CVAT_USER, CVAT_PASS),
            timeout=10
        )
        response.raise_for_status()
        task_info = response.json()
        task_name = task_info.get("name", f"task_{task_id}")
        video_name = task_name.replace(f"_{split}", "")
    except Exception as e:
        log.warning(f"⚠️ Could not fetch task name from CVAT: {e}. Using Task ID instead.")
        video_name = f"task_{task_id}"

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    export_filename = f"{video_name}.zip"
    export_path = LABELS_DIR / export_filename
    
    dump_cmd = (
        f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
        f'--auth {CVAT_USER}:{CVAT_PASS} dump {task_id} "{export_path}" --format "MOT 1.1"'
    )
    run_cmd(dump_cmd)
    
    log.info(f"☁️ Sending the corrected annotation to Hugging Face as {export_filename}...")
    path_in_repo = f"labels/{split}/{batch_name}/{export_filename}"
    
    api.upload_file(
        path_or_fileobj=str(export_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload manually validated MOT annotations for {video_name}"
    )
    
    log.info(f"✅ Success! The verified archive has been saved to HF: {path_in_repo}")