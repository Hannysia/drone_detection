from io import BytesIO
from pathlib import Path
import zipfile
import numpy as np
from src.processing.video import get_video_metadata


def prepare_trackeval_sequence(
    video_path: Path,
    zip_path: Path,
    seq_gt_dir: Path,
    seqinfo_path: Path,
) -> None:
    """Extracts ground truth from zip, standardizes category ID to 1, and writes metadata."""
    width, height, fps, length = get_video_metadata(str(video_path))

    seq_name = video_path.stem
    seqinfo_content = (
        f"[Sequence]\nname={seq_name}\nimDir=img1\n"
        f"frameRate={int(fps)}\nseqLength={length}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n"
    )

    with zipfile.ZipFile(zip_path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and Path(entry.filename).name == "gt.txt"
        ]
        if len(entries) != 1:
            raise ValueError(
                f"{zip_path}: expected exactly one gt.txt, found {len(entries)}"
            )
        contents = archive.read(entries[0])

    target = seq_gt_dir / "gt.txt"

    if contents.strip():
        data = np.loadtxt(BytesIO(contents), delimiter=",", ndmin=2)
        if data.shape[1] != 9:
            raise ValueError(f"{zip_path}: expected 9 columns in gt.txt")

        data[:, 7] = 1
        np.savetxt(
            target,
            data,
            fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f",
        )
    else:
        target.write_bytes(b"")

    seqinfo_path.write_text(seqinfo_content, encoding="utf-8")


def compile_kaggle_mot_dataset(
    local_hf_dir: Path,
    mot_dataset_dir: Path,
    seqmaps_dir: Path,
    video_extensions: tuple[str],
) -> None:
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

                prepare_trackeval_sequence(
                    video_path,
                    zip_path,
                    seq_gt_dir,
                    seqinfo_path,
                )
            else:
                print(f"Warning: No matching zip archive found for video {video_path.name}")

        with open(seqmaps_dir / f"{trackeval_split_name}.txt", "w", encoding="utf-8") as f:
            f.write("name\n" + "\n".join(matched_seqs))

    print("Dataset compilation for TrackEval complete.")