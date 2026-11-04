import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.processing.manifests import CATEGORY_DIRS, read_manifest

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

        for image_path in self.image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if category == "positive":
                label_path = image_path.with_suffix(".txt")
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)

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
    batch_size: int = 64,
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