import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from src.config import EMBEDDINGS_DIR

log = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class DroneEmbeddingDataset(Dataset):
    def __init__(self, raw_dir: Path, subdir_name: str = "RAW_POSITIVES"):
        self.raw_dir = Path(raw_dir)
        self.subdir_name = subdir_name
        self.target_dir = self.raw_dir / subdir_name
        self.image_paths = sorted(list(self.target_dir.glob("*.jpg")))
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def parse_yolo_bbox(self, txt_path: Path) -> list[float] | None:
        """Parses first bounding box coordinates from a YOLO annotation file."""
        if not txt_path.exists():
            return None
        try:
            with open(txt_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return None
            parts = lines[0].strip().split()
            if len(parts) == 5:
                return list(map(float, parts[1:]))
        except Exception as e:
            log.warning(f"Failed to parse YOLO labels for file {txt_path.name}: {e}")
        return None

    def get_adaptive_crop(self, image: Image.Image, bbox: list[float], bbox_area_ratio: float) -> Image.Image:
        """Crops drone silhouette with dynamic padding based on target scale."""
        img_w, img_h = image.size
        cx, cy, w, h = bbox
        
        # Apply larger padding to smaller targets to capture context
        if bbox_area_ratio < 0.01:
            padding_factor = 4.0
        elif bbox_area_ratio < 0.05:
            padding_factor = 2.0
        else:
            padding_factor = 1.2
            
        crop_size = max(w * img_w, h * img_h) * padding_factor
        crop_size = max(crop_size, 64)  # Prevent tiny crops
        
        cx_px, cy_px = cx * img_w, cy * img_h
        
        x1 = max(0, int(cx_px - crop_size / 2))
        y1 = max(0, int(cy_px - crop_size / 2))
        x2 = min(img_w, int(cx_px + crop_size / 2))
        y2 = min(img_h, int(cy_px + crop_size / 2))
        
        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        
        if self.subdir_name == "RAW_NEGATIVES":
            bbox = None
        else:
            txt_path = img_path.with_suffix(".txt")
            bbox = self.parse_yolo_bbox(txt_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
            
        if bbox is not None:
            w, h = bbox[2], bbox[3]
            bbox_area_ratio = w * h
            crop_img = self.get_adaptive_crop(image, bbox, bbox_area_ratio)
        else:
            bbox_area_ratio = 0.0
            crop_img = Image.new("RGB", (224, 224))
            
        full_tensor = IMAGE_TRANSFORM(image)
        crop_tensor = IMAGE_TRANSFORM(crop_img)
        
        return full_tensor, crop_tensor, torch.tensor(bbox_area_ratio, dtype=torch.float32), img_path.name

def generate_and_save_embeddings(
    dataset_input_dir: Path, 
    model: nn.Module, 
    split_name: str = "train", 
    subdir_name: str = "RAW_POSITIVES",
    batch_size: int = 64,
    device: torch.device = torch.device("cpu")
) -> Path:
    """Generates dual-weighted embeddings using pre-trained DINOv2 model and saves them."""
    print("\n" + "="*50)
    print(f"GENERATING EMBEDDINGS FOR: {split_name.upper()} ({subdir_name})")
    print(f"Reading directly from: {dataset_input_dir}")
    print("="*50)
    
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset = DroneEmbeddingDataset(dataset_input_dir, subdir_name=subdir_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    embeddings_dict = {}
    
    print("\nExtracting embeddings via DINOv2 model...")
    progress_bar = tqdm(dataloader, desc="DINOv2 Inference", total=len(dataloader))
    
    with torch.no_grad():
        for full_imgs, crop_imgs, area_ratios, filenames in progress_bar:
            full_imgs = full_imgs.to(device)
            crop_imgs = crop_imgs.to(device)
            area_ratios = area_ratios.to(device)
            
            emb_full = model(full_imgs) 
            emb_crop = model(crop_imgs) 
            
            emb_full = F.normalize(emb_full, p=2, dim=1)
            emb_crop = F.normalize(emb_crop, p=2, dim=1)
            
            w_full = torch.where(area_ratios == 0.0, torch.ones_like(area_ratios), area_ratios).unsqueeze(1)
            w_crop = torch.where(area_ratios == 0.0, torch.zeros_like(area_ratios), 1.0 - area_ratios).unsqueeze(1)
            
            emb_full_weighted = emb_full * w_full
            emb_crop_weighted = emb_crop * w_crop
            
            final_embeddings = torch.cat([emb_full_weighted, emb_crop_weighted], dim=1)
            final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
            final_embeddings = final_embeddings.cpu().numpy()

            for name, emb in zip(filenames, final_embeddings):
                embeddings_dict[name] = emb
                
    suffix = "positives" if subdir_name == "RAW_POSITIVES" else "negatives"
    output_file_name = f"{split_name}_{suffix}_embeddings.pt"
    output_file_path = EMBEDDINGS_DIR / output_file_name
    
    print(f"\nSaving embeddings dictionary to {output_file_path}...")
    torch.save(embeddings_dict, output_file_path)
    
    return output_file_path