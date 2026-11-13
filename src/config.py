import os
from pathlib import Path

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v']
SPLITS = ["train", "val", "test"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
HISTORICAL_MANIFEST = MANIFESTS_DIR / "historical" / "final.csv"

IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

EMBEDDING_MODEL_NAME = "dinov2_vitb14"
EMBEDDING_MODEL_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EMBEDDING_BATCH_SIZE = 64

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    
    user_secrets = UserSecretsClient()
    REPO_ID = user_secrets.get_secret("REPO_ID")
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    
    KAGGLE_INPUT_DIR = Path("/kaggle/input")
    parent_dir = KAGGLE_INPUT_DIR / "datasets" / "hannusiaknysh"
    
    IMG_DATASET_NAME = "uav-detection-dataset"
    
    if parent_dir.exists():
        attached_datasets = [d.name for d in parent_dir.iterdir() if d.is_dir()]
        if attached_datasets:
            IMG_DATASET_NAME = attached_datasets[0]
            
    EXTRACTED_DATA_DIR = parent_dir / IMG_DATASET_NAME / "extracted_dataset"
    WORKING_DIR = Path("/kaggle/working")
    FRAME_BASELINE_DIR = WORKING_DIR / "frame_baseline"
    OUTPUT_DIR = WORKING_DIR / "runs" / "detect" 
    EMBEDDINGS_DIR = Path("/kaggle/working/embeddings")
    
    MOT_DATASET_DIR = WORKING_DIR / "drone_mot_dataset"
    SEQMAPS_DIR = MOT_DATASET_DIR / "seqmaps"
    YOLO_DETECTIONS_OUT = WORKING_DIR / "yolo_detections"
    YOLO_MODEL_PATH_PT = WORKING_DIR / "drone_dataset/models/yolo26n_final_model_best.pt"
    YOLO_MODEL_PATH_ENGINE = WORKING_DIR / "drone_dataset/models/yolo26n_final_model_best.engine"

    TRACKERS_OUT_DIR = WORKING_DIR / "trackers"
    LOCAL_HF_DATA_DIR = WORKING_DIR / "drone_dataset"

else:
    from dotenv import load_dotenv
    load_dotenv()

    _PATH_ENV = {
        "LOCAL_BASE_DIR": "LOCAL_BASE_DIR",
        "EMBEDDINGS_DIR": "EMBEDDINGS_DIR",
        "EXTRACTED_DATA_DIR": "EXTRACTED_DATA_PATH",
        "TRASH_DIR": "TRASH_DIR",
        "MT_TRASH_DIR": "MT_TRASH_DIR",
        "SOURCE_ROOT": "SOURCE_ROOT",
        "CLASSIFIED_DIR": "CLASSIFIED_DIR",
        "KEY_JSON_PATH": "KEY_JSON_PATH",
        "RENAMED_DIR": "RENAMED_DIR",
        "SPLIT_OUTPUT_DIR": "SPLIT_OUTPUT_DIR",
        "LABELS_DIR": "LABELS_DIR",
        "TRACKEVAL_DIR": "TRACKEVAL_DIR",
        "DUPLICATES_DIR": "DUPLICATES_DIR",
    }

    _BASE_CHILDREN = {
        "FRAME_BASELINE_DIR": "frame_baseline",
        "WORKING_DIR": "working",
        "OUTPUT_DIR": "extracted_dataset",
        "LOG_FILE": "processed_videos.log",
        "TMP_DIR": "tmp",
    }

    def __getattr__(name: str) -> Path:
        child = _BASE_CHILDREN.get(name)
        key = "LOCAL_BASE_DIR" if child is not None else _PATH_ENV.get(name)

        if key is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        value = os.getenv(key)
        if not value or not value.strip():
            raise ValueError(f"Set {key} in the environment or .env to use {name}")

        path = Path(value)
        return path / child if child is not None else path

    REPO_ID = os.getenv("REPO_ID")
    HF_TOKEN = os.getenv("HF_TOKEN")

    HASH_THRESHOLD = 5
    SAVE_DEBUG_FRAMES = False
    FORCE_CLASS_ID = os.getenv("FORCE_CLASS_ID", "0")

    CVAT_HOST = os.getenv("CVAT_HOST", "localhost")
    CVAT_PORT = os.getenv("CVAT_PORT", "8080")
    CVAT_USER = os.getenv("CVAT_USER")
    CVAT_PASS = os.getenv("CVAT_PASS")