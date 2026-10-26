import logging
import re
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_download

from src.config import (
    REPO_ID, CVAT_HOST, CVAT_PORT, LABELS_DIR, CVAT_USER, CVAT_PASS, HF_TOKEN, VIDEO_EXTENSIONS, SPLITS
)
from src.utils.shell import run_cmd

log = logging.getLogger(__name__)
api = HfApi(token=HF_TOKEN)

ANNOTATION_SOURCES = {
    "sam3": ("cvat_batches", "_mot.zip"),
    "labels": ("labels", ".zip"),
}

CURATED_LABELS_ROOT = "labels_curated"

def curated_annotation_path(
    version: str,
    split: str,
    batch_name: str,
    video_name: str,
) -> str:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError(f"Invalid dataset version: {version}")

    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")

    for name in (batch_name, video_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid name: {name}")

    return (
        f"{CURATED_LABELS_ROOT}/{version}/"
        f"{split}/{batch_name}/{video_name}.zip"
    )


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


def push_batch_to_hf(
    task_id: str,
    split: str,
    batch_name: str,
    version: str,
) -> None:
    log.info(f"Exporting CVAT annotations for Task {task_id}")

    response = requests.get(
        f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
        auth=(CVAT_USER, CVAT_PASS),
        timeout=10,
    )
    response.raise_for_status()

    task_name = response.json()["name"]
    suffix = f"_{split}"

    if not task_name.endswith(suffix):
        raise ValueError(f"Unexpected CVAT task name: {task_name}")

    video_name = task_name[:-len(suffix)]

    path_in_repo = curated_annotation_path(
        version=version,
        split=split,
        batch_name=batch_name,
        video_name=video_name,
    )

    if api.file_exists(repo_id=REPO_ID, filename=path_in_repo, repo_type="dataset",):
        raise FileExistsError(f"Annotation already exists on HF: {path_in_repo}")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    export_filename = f"{video_name}.zip"
    export_path = LABELS_DIR / export_filename

    dump_cmd = (
        f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
        f'--auth {CVAT_USER}:{CVAT_PASS} dump {task_id} '
        f'"{export_path}" --format "MOT 1.1"'
    )

    run_cmd(dump_cmd)

    if not export_path.is_file() or export_path.stat().st_size == 0:
        raise RuntimeError(
            f"CVAT export was not created: {export_path}"
        )

    result = api.upload_file(
        path_or_fileobj=str(export_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=(
            f"Upload curated MOT annotations for "
            f"{video_name} ({version})"
        ),
    )

    log.info(f"Saved to HF: {path_in_repo}")