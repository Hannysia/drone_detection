import os
from pathlib import Path

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v']
SPLITS = ["train", "val", "test"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
HISTORICAL_MANIFEST = MANIFESTS_DIR / "historical" / "final.csv"

IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

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

    IOU_THRESHOLD = 0.95
    THETA_1_wv_pos = 0.10 
    THETA_2_wv_pos = 0.538
    THETA_1_wv_neg = 0.020
    THETA_2_wv_neg = 0.025
    HASH_THRESHOLD = 5
    MIN_BBOX_AREA = 256.0
    FORCE_CLASS_ID = "0"

    REPO_ID = os.getenv("REPO_ID")
    HF_TOKEN = os.getenv("HF_TOKEN")
    LOCAL_BASE_DIR = Path(os.getenv("LOCAL_BASE_DIR"))

    WORKING_DIR = LOCAL_BASE_DIR / "working"


    OUTPUT_DIR = LOCAL_BASE_DIR / "extracted_dataset"
    LOG_FILE = LOCAL_BASE_DIR / "processed_videos.log"
    TMP_DIR = LOCAL_BASE_DIR / "tmp"

    EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR"))

    EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
    TRASH_DIR = Path(os.getenv("TRASH_DIR"))

    MIN_BBOX_AREA = float(os.getenv("MIN_BBOX_AREA", 256.0))
    MT_TRASH_DIR = Path(os.getenv("MT_TRASH_DIR"))


    SAMPLE_EVERY = {
        ("val",  "positives"): 4,
        ("val",  "negatives"): 1,
        ("test", "positives"): 3,
        ("test", "negatives"): 1,
    }

    CAP_PER_VIDEO = {
        ("val",  "positives"): None,
        ("val",  "negatives"): 25,
        ("test", "positives"): None,
        ("test", "negatives"): 45,
    }

    SOURCE_ROOT = Path(os.getenv("SOURCE_ROOT"))
    CLASSIFIED_DIR = Path(os.getenv("CLASSIFIED_DIR"))
    SAVE_DEBUG_FRAMES = False
    KEY_JSON_PATH = Path(os.getenv("KEY_JSON_PATH"))
    RENAMED_DIR = Path(os.getenv("RENAMED_DIR"))
    SPLIT_OUTPUT_DIR = Path(os.getenv("SPLIT_OUTPUT_DIR"))
    CVAT_HOST = os.getenv("CVAT_HOST", "localhost")
    CVAT_PORT = os.getenv("CVAT_PORT", "8080")
    LABELS_DIR = Path(os.getenv("LABELS_DIR"))
    CVAT_USER = os.getenv("CVAT_USER")
    CVAT_PASS = os.getenv("CVAT_PASS")
    TRACKEVAL_DIR = Path(os.getenv("TRACKEVAL_DIR"))
    DUPLICATES_DIR = Path(os.getenv("DUPLICATES_DIR"))
    FORCE_CLASS_ID = os.getenv("FORCE_CLASS_ID", "0")