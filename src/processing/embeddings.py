import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDINGS_DIR,
    MANIFESTS_DIR,
    PROJECT_ROOT,
    HF_TOKEN,
)
from src.processing.manifests import CATEGORY_DIRS, read_manifest
from src.utils.io import check_extraction_source

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class DroneEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        category: str,
    ):
        if category not in CATEGORY_DIRS:
            raise ValueError(f"Invalid category: {category}")

        self.category = category
        rows = read_manifest(manifest_path)
        self.image_paths = sorted(
            dataset_root / "train" / CATEGORY_DIRS[category] / row["filename"]
            for row in rows
            if row["split"] == "train"
            and row["category"] == category
            and not row["excluded_at"]
        )

        if not self.image_paths:
            raise ValueError(f"No active train frames for {category}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def parse_yolo_bbox(self, txt_path: Path) -> list[float]:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()

        if len(lines) != 1 or len(lines[0].split()) != 5:
            raise ValueError(f"Expected exactly one YOLO box: {txt_path}")

        bbox = list(map(float, lines[0].split()[1:]))
        cx, cy, width, height = bbox

        if (
            not all(math.isfinite(value) for value in bbox)
            or not 0 <= cx <= 1
            or not 0 <= cy <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            raise ValueError(f"Invalid YOLO box: {txt_path}")

        return bbox

    def get_adaptive_crop(
        self,
        image: Image.Image,
        bbox: list[float],
        bbox_area_ratio: float,
    ) -> Image.Image:
        img_w, img_h = image.size
        cx, cy, w, h = bbox

        if bbox_area_ratio < 0.01:
            padding_factor = 4.0
        elif bbox_area_ratio < 0.05:
            padding_factor = 2.0
        else:
            padding_factor = 1.2

        crop_size = max(w * img_w, h * img_h) * padding_factor
        crop_size = max(crop_size, 64)

        cx_px, cy_px = cx * img_w, cy * img_h

        x1 = max(0, int(cx_px - crop_size / 2))
        y1 = max(0, int(cy_px - crop_size / 2))
        x2 = min(img_w, int(cx_px + crop_size / 2))
        y2 = min(img_h, int(cy_px + crop_size / 2))

        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.category == "positive":
            bbox = self.parse_yolo_bbox(image_path.with_suffix(".txt"))
            area = bbox[2] * bbox[3]
            crop = self.get_adaptive_crop(image, bbox, area)
        else:
            area = 0.0
            crop = Image.new("RGB", (224, 224))

        return (
            IMAGE_TRANSFORM(image),
            IMAGE_TRANSFORM(crop),
            torch.tensor(area, dtype=torch.float32),
            image_path.name,
        )


def generate_and_save_embeddings(
    dataset_root: Path,
    manifest_path: Path,
    model: nn.Module,
    category: str,
    output_path: Path,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 2,
) -> Path:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    dataset = DroneEmbeddingDataset(dataset_root, manifest_path, category)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = {}
    model.eval()

    with torch.inference_mode():
        for full, crop, areas, names in tqdm(
            loader,
            desc=f"train/{category}",
        ):
            areas = areas.to(device).unsqueeze(1)

            full_embeddings = F.normalize(
                model(full.to(device)), p=2, dim=1,
            )
            crop_embeddings = F.normalize(
                model(crop.to(device)), p=2, dim=1,
            )

            full_weights = torch.where(areas == 0, 1.0, areas)
            crop_weights = torch.where(areas == 0, 0.0, 1.0 - areas)

            combined = F.normalize(
                torch.cat([
                    full_embeddings * full_weights,
                    crop_embeddings * crop_weights,
                ], dim=1),
                p=2,
                dim=1,
            )

            if (
                not torch.isfinite(combined).all()
                or not torch.allclose(
                    combined.norm(dim=1),
                    torch.ones(len(names), device=device),
                )
            ):
                raise ValueError("Model produced invalid embeddings")

            embeddings.update(zip(names, combined.cpu().numpy()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with NamedTemporaryFile(
            dir=output_path.parent,
            prefix=".embeddings-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(embeddings, stream)

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path


def _bind_run(output_dir: Path, metadata: dict) -> None:
    metadata_path = output_dir / "run.json"

    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing embeddings belong to a different run")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        raise ValueError("Output directory has files without run.json")

    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".run-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary_path, metadata_path)
        except FileExistsError:
            _bind_run(output_dir, metadata)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate train embeddings from a manifest",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--category",
        choices=tuple(CATEGORY_DIRS),
    )
    mode.add_argument("--publish-only", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check inputs or saved embeddings without generating or publishing",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    source = json.loads(
        (MANIFESTS_DIR / "sources" / "hf.json").read_text(
            encoding="utf-8",
        )
    )
    
    if args.publish_only:
        files = validate_embeddings(args.dataset_root, source)

        if not args.check_only:
            publish_embeddings(
                files,
                repo_id=source["repo_id"],
                version=source["annotation_version"],
                token=HF_TOKEN,
            )

        return

    check_extraction_source(args.dataset_root, source)

    version = source["annotation_version"]
    manifest = args.dataset_root / "manifest.csv"
    rows = read_manifest(manifest)
    expected = read_manifest(
        MANIFESTS_DIR / "baseline" / version / "main.csv",
    )

    if rows != expected:
        raise ValueError(
            "Uploaded manifest differs from the project baseline",
        )

    dataset = DroneEmbeddingDataset(
        args.dataset_root, manifest, args.category,
    )
    print(f"train/{args.category}: {len(dataset)}", flush=True)

    if args.check_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Enable a GPU accelerator before generating embeddings",
        )

    output_dir = EMBEDDINGS_DIR / version
    output_path = output_dir / f"train_{args.category}s_embeddings.pt"

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    metadata = {
        "source": source,
        "manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "code_revision": revision,
        "model": EMBEDDING_MODEL_NAME,
        "model_code_revision": EMBEDDING_MODEL_REVISION,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "batch_size": args.batch_size,
        "input_counts": dict(Counter(
            row["category"]
            for row in rows
            if row["split"] == "train" and not row["excluded_at"]
        )),
    }
    _bind_run(output_dir, metadata)

    device = torch.device("cuda")

    print(f"Loading {EMBEDDING_MODEL_NAME} on CUDA...", flush=True)
    model = torch.hub.load(
        f"facebookresearch/dinov2:{EMBEDDING_MODEL_REVISION}",
        EMBEDDING_MODEL_NAME,
        trust_repo=True,
    ).to(device)

    try:
        generate_and_save_embeddings(
            dataset_root=args.dataset_root,
            manifest_path=manifest,
            model=model,
            category=args.category,
            output_path=output_path,
            batch_size=args.batch_size,
            device=device,
        )
        print("Saved:", output_path)
    finally:
        del model
        torch.cuda.empty_cache()

def validate_embeddings(
    dataset_root: Path,
    source: dict,
) -> list[Path]:
    check_extraction_source(dataset_root, source)

    version = source["annotation_version"]
    manifest = dataset_root / "manifest.csv"
    rows = read_manifest(manifest)
    expected_rows = read_manifest(
        MANIFESTS_DIR / "baseline" / version / "main.csv",
    )

    if rows != expected_rows:
        raise ValueError(
            "Uploaded manifest differs from the project baseline",
        )

    output_dir = EMBEDDINGS_DIR / version
    metadata_path = output_dir / "run.json"
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8"),
    )
    counts = dict(Counter(
        row["category"]
        for row in rows
        if row["split"] == "train" and not row["excluded_at"]
    ))

    if (
        metadata["source"] != source
        or metadata["manifest_sha256"]
        != hashlib.sha256(manifest.read_bytes()).hexdigest()
        or metadata["input_counts"] != counts
    ):
        raise ValueError(
            "Embedding metadata does not match the dataset",
        )

    files = [metadata_path]
    dimension = None

    for category in CATEGORY_DIRS:
        path = output_dir / f"train_{category}s_embeddings.pt"
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        expected = {
            row["filename"]
            for row in rows
            if row["split"] == "train"
            and row["category"] == category
            and not row["excluded_at"]
        }

        if (
            not isinstance(data, dict)
            or not expected
            or set(data) != expected
        ):
            raise ValueError(
                f"Embedding filenames mismatch: {path}",
            )

        for name, vector in data.items():
            if (
                dimension is None
                and isinstance(vector, np.ndarray)
                and vector.ndim == 1
            ):
                dimension = vector.size

            if (
                not isinstance(vector, np.ndarray)
                or vector.ndim != 1
                or vector.size != dimension
                or not vector.size
                or vector.size % 2
                or not np.issubdtype(vector.dtype, np.floating)
                or not np.isfinite(vector).all()
                or not np.isclose(
                    np.linalg.norm(vector),
                    1.0,
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise ValueError(
                    f"Invalid embedding: {path.name}/{name}",
                )

            if (
                category == "negative"
                and np.count_nonzero(vector[dimension // 2:])
            ):
                raise ValueError(
                    f"Nonzero crop component for negative frame: {name}",
                )

        print(
            f"Verified train/{category}: {len(data)}, "
            f"dimension={dimension}",
            flush=True,
        )
        files.append(path)
        del data

    return files


def publish_embeddings(
    files: list[Path],
    *,
    repo_id: str,
    version: str,
    token: str,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    if not files or len({path.name for path in files}) != len(files):
        raise ValueError(
            "Publication requires files with unique names",
        )

    if not token:
        raise ValueError("HF_TOKEN is missing")

    api = HfApi(token=token)
    prefix = f"embeddings/{version}"
    targets = {f"{prefix}/{path.name}" for path in files}

    head = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
    ).sha

    existing = set(api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=head,
    ))

    if targets & existing:
        raise FileExistsError(
            f"HF files already exist: {sorted(targets & existing)}",
        )

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        parent_commit=head,
        commit_message=(
            f"feat(data): publish validated {version} train embeddings"
        ),
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in files
        ],
    )

    published = set(api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=commit.oid,
    ))

    if not targets <= published:
        raise RuntimeError(
            f"Published files missing at {commit.commit_url}",
        )

    print("Published:", commit.commit_url, flush=True)
    return commit.commit_url

if __name__ == "__main__":
    main()