
import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    REPO_ID,
    HF_TOKEN,
    VIDEO_EXTENSIONS,
    SPLITS,
    PROJECT_ROOT,
    FRAME_BASELINE_DIR,
    EXTRACTED_DATA_DIR,
)
from src.utils.io import (
    get_processed_list,
    mark_as_processed,
    cleanup_partial_files,
    get_extraction_paths,
)
from src.processing.extractor import (
    download_and_extract_assets,
    parse_annotations,
    extract_frames_and_labels,
    select_annotation_zips,
)


def main(
    target_split: str | None = None,
    target_video: str | None = None,
) -> None:

    output_dir, tmp_dir, log_file = get_extraction_paths(
        baseline_dir=FRAME_BASELINE_DIR,
        historical_dir=EXTRACTED_DATA_DIR,
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extraction output: {output_dir}")
    print(f"Temporary files: {tmp_dir}")
    print(f"Progress log: {log_file}")

    api = HfApi(token=HF_TOKEN)

    source_path = PROJECT_ROOT / "manifests" / "sources" / "hf.json"

    with source_path.open("r", encoding="utf-8") as file:
        source = json.load(file)

    if source["repo_id"] != REPO_ID:
        raise ValueError(
            "Hugging Face repository ID does not match "
            "the pinned source configuration"
        )

    if source["repo_type"] != "dataset":
        raise ValueError("Expected a Hugging Face dataset repository")

    revision = source["revision"]

    print(f"Fetching files from HF revision: {revision}")

    repo_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
    )

    processed_videos = get_processed_list(log_file)

    for split in SPLITS:

        if target_split is not None and split != target_split:
            continue

        print("\n" + "="*80)
        print(f"STARTING SPLIT: {split.upper()}")
        print("="*80)

        split_pos_dir = output_dir / split / "RAW_POSITIVES"
        split_neg_dir = output_dir / split / "RAW_NEGATIVES"
        split_tmp_dir = tmp_dir / split

        split_pos_dir.mkdir(parents=True, exist_ok=True)
        split_neg_dir.mkdir(parents=True, exist_ok=True)
        split_tmp_dir.mkdir(parents=True, exist_ok=True)

        zip_tasks = select_annotation_zips(
            repo_files=repo_files,
            split=split,
            video_name=target_video,
        )

        for zip_path in zip_tasks:
            video_base_name = Path(zip_path).stem
            
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
                message = f"Video not found: {split}/{video_base_name}"

                if target_video is not None:
                    raise FileNotFoundError(message)

                print(message)
                continue

            cleanup_partial_files(video_base_name, split_pos_dir, split_neg_dir)

            print(f"\nProcessing: {video_base_name}")
            
            try:

                local_video, gt_temp_path = download_and_extract_assets(
                    REPO_ID,
                    HF_TOKEN,
                    video_path_hf,
                    zip_path,
                    split_tmp_dir,
                    video_base_name,
                    revision=revision,
                )

                annotations = parse_annotations(gt_temp_path)
                
                extract_frames_and_labels(
                    local_video, annotations, video_base_name, split_pos_dir, split_neg_dir
                )
                
                mark_as_processed(log_file, video_base_name)
                
                local_video.unlink(missing_ok=True)
                gt_temp_path.unlink(missing_ok=True)
                (split_tmp_dir / Path(zip_path).name).unlink(missing_ok=True)
                
                print(f"✅ Success: {video_base_name}")

            except Exception as e:
                print(f"ERROR processing {video_base_name}: {e}")

                if target_video is not None:
                    raise

    print("\n" + "="*80)
    print("PROCESS FINISHED")
    print(f"Check {log_file} for the list of completed videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a pinned HF dataset revision."
    )

    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument(
        "--video",
        help="Video filename without extension",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every video explicitly",
    )

    args = parser.parse_args()

    if args.all:
        if args.split is not None or args.video is not None:
            parser.error("--all cannot be combined with --split/--video")
    elif args.split is None or args.video is None:
        parser.error("Provide --split and --video, or use --all")

    main(
        target_split=args.split,
        target_video=args.video,
    )
