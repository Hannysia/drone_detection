from tempfile import TemporaryDirectory
import logging
import re
from pathlib import Path
import requests
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import json
import zipfile

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



def find_existing_cvat_task_ids(task_name: str) -> list[int]:
    task_ids = []
    page = 1

    while True:
        response = requests.get(
            f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks",
            auth=(CVAT_USER, CVAT_PASS),
            headers={"Accept": "application/vnd.cvat+json"},
            params={
                "name": task_name,
                "page": page,
                "page_size": 100,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if (
            not isinstance(data, dict)
            or not isinstance(data.get("results"), list)
            or "next" not in data
        ):
            raise ValueError("Unexpected CVAT tasks API response")

        for task in data["results"]:
            if (
                not isinstance(task, dict)
                or "id" not in task
                or "name" not in task
            ):
                raise ValueError("Invalid CVAT task entry")

            if task["name"] == task_name:
                task_ids.append(task["id"])

        if not data["next"]:
            break

        page += 1

    return task_ids



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

        task_name = f"{base_name}_{split}"
        existing_ids = find_existing_cvat_task_ids(task_name)

        if existing_ids:
            log.warning(
                "⏭️ Skipping %s: CVAT task already exists (ID: %s).",
                task_name,
                ", ".join(map(str, existing_ids)),
            )
            log.warning(
                "Check its annotations if a previous import was interrupted."
            )
            continue
        
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
        
        create_cmd = (
            f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
            f'--auth {CVAT_USER}:{CVAT_PASS} create "{task_name}" '
            f'--labels "[{{\\"name\\": \\"pedestrian\\\"}}, {{\\"name\\": \\"drone\\\"}}]" '
            f'local "{local_video_path}"'
        )
        output = run_cmd(create_cmd)
        
        match = re.search(r'(?i)ID[\s:]*(\d+)', output)
        if not match:
            raise RuntimeError(
                f"Could not determine Task ID for {task_name}. "
                f"Check CVAT before retrying. "
                f"CLI output: {output.strip()}"
            )
            
        task_id = match.group(1)
        log.info(f"✅ Task created! Task ID: {task_id}")

        log.info(f"📦 Uploading {annotation_source} annotations to task...")
        upload_cmd = (
            f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
            f'--auth {CVAT_USER}:{CVAT_PASS} upload {task_id} "{local_zip_path}" --format "MOT 1.1"'
        )
        run_cmd(upload_cmd)
        
        log.info(f"🎉 Done: http://{CVAT_HOST}:{CVAT_PORT}/tasks/{task_id}")


def build_review_metadata(
    meta: dict,
    video_name: str,
    task_id: str,
    version: str,
) -> dict:
    required = (
        "size",
        "start_frame",
        "stop_frame",
        "frame_filter",
        "included_frames",
        "deleted_frames",
    )

    missing = [key for key in required if key not in meta]

    if missing:
        raise ValueError(
            f"Missing CVAT metadata fields: {missing}"
        )

    if type(meta["size"]) is not int or meta["size"] <= 0:
        raise ValueError("Invalid CVAT frame count")

    if (
        type(meta["start_frame"]) is not int
        or type(meta["stop_frame"]) is not int
        or meta["start_frame"] < 0
        or meta["stop_frame"] < meta["start_frame"]
    ):
        raise ValueError("Invalid CVAT frame range")

    deleted = meta["deleted_frames"]

    if (
        not isinstance(deleted, list)
        or any(
            type(index) is not int or index < 0
            for index in deleted
        )
        or len(deleted) != len(set(deleted))
    ):
        raise ValueError("Invalid CVAT deleted_frames")

    included = meta["included_frames"]

    if included is not None and (
        not isinstance(included, list)
        or any(
            type(index) is not int or index < 0
            for index in included
        )
        or len(included) != len(set(included))
    ):
        raise ValueError("Invalid CVAT included_frames")

    if not isinstance(meta["frame_filter"], str):
        raise ValueError("Invalid CVAT frame_filter")

    return {
        "video_name": video_name,
        "task_id": int(task_id),
        "version": version,
        "size": meta["size"],
        "start_frame": meta["start_frame"],
        "stop_frame": meta["stop_frame"],
        "frame_filter": meta["frame_filter"],
        "included_frames": included,
        "deleted_frames": sorted(deleted),
    }



def push_batch_to_hf(
    task_id: str,
    split: str,
    batch_name: str,
    version: str,
) -> None:
    response = requests.get(
        f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
        auth=(CVAT_USER, CVAT_PASS),
        headers={"Accept": "application/vnd.cvat+json"},
        timeout=30,
    )
    response.raise_for_status()

    task_name = response.json()["name"]
    suffix = f"_{split}"

    if not task_name.endswith(suffix) or len(task_name) <= len(suffix):
        raise ValueError(f"Unexpected CVAT task name: {task_name}")

    video_name = task_name[:-len(suffix)]

    repo_zip = curated_annotation_path(
        version,
        split,
        batch_name,
        video_name,
    )

    repo_review = (
        repo_zip.removesuffix(".zip") + ".review.json"
    )

    head_revision = api.dataset_info(
        repo_id=REPO_ID
    ).sha

    
    archive_on_hf = api.file_exists(
        repo_id=REPO_ID,
        filename=repo_zip,
        repo_type="dataset",
        revision=head_revision,
    )

    review_on_hf = api.file_exists(
        repo_id=REPO_ID,
        filename=repo_review,
        repo_type="dataset",
        revision=head_revision,
    )

    if archive_on_hf and review_on_hf:
        log.info(
            "Already published on HF: %s and %s. Skipping.",
            repo_zip,
            repo_review,
        )
        return

    if archive_on_hf or review_on_hf:
        raise RuntimeError(
            f"Incomplete HF export for {video_name}: "
            f"MOT exists={archive_on_hf}, "
            f"review exists={review_on_hf}. "
            f"Check HF files before retrying."
        )


    meta_response = requests.get(
        f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}/data/meta",
        auth=(CVAT_USER, CVAT_PASS),
        headers={"Accept": "application/vnd.cvat+json"},
        timeout=30,
    )
    meta_response.raise_for_status()

    review_data = build_review_metadata(
        meta_response.json(),
        video_name,
        task_id,
        version,
    )

    export_dir = (
        LABELS_DIR / "exports" / version / split / batch_name
    )

    archive_path = export_dir / f"{video_name}.zip"
    review_path = export_dir / f"{video_name}.review.json"

    export_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        prefix=".cvat_push_",
        dir=export_dir,
    ) as staging_dir:
        staging_archive = Path(staging_dir) / archive_path.name
        staging_review = Path(staging_dir) / review_path.name

        dump_cmd = (
            f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} '
            f'--auth {CVAT_USER}:{CVAT_PASS} dump {task_id} '
            f'"{staging_archive}" --format "MOT 1.1"'
        )

        run_cmd(dump_cmd)

        if not zipfile.is_zipfile(staging_archive):
            raise ValueError(
                f"CVAT export is not a valid ZIP: {staging_archive}"
            )

        with zipfile.ZipFile(staging_archive) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError(
                    f"Corrupted CVAT ZIP: {staging_archive}"
                )

        staging_review.write_text(
            json.dumps(
                review_data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        staging_archive.replace(archive_path)
        staging_review.replace(review_path)

    result = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        parent_commit=head_revision,
        commit_message=(
            f"Upload curated MOT and frame exclusions "
            f"for {video_name} ({version})"
        ),
        operations=[
            CommitOperationAdd(
                path_in_repo=repo_zip,
                path_or_fileobj=str(archive_path),
            ),
            CommitOperationAdd(
                path_in_repo=repo_review,
                path_or_fileobj=str(review_path),
            ),
        ],
    )

    log.info("Saved MOT: %s", repo_zip)
    log.info("Saved frame exclusions: %s", repo_review)
    log.info(
        "Deleted frames: %d",
        len(review_data["deleted_frames"]),
    )
    log.info("HF commit: %s", result.oid)
