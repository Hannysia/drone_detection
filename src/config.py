import os
import sys
from pathlib import Path

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v']
SPLITS = ["train", "val", "test"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
HISTORICAL_MANIFEST = MANIFESTS_DIR / "historical" / "final.csv"

IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''
IS_COLAB = not IS_KAGGLE and (
    os.environ.get('COLAB_RELEASE_TAG', '') != ''
    or os.environ.get('COLAB_GPU', '') != ''
    or 'google.colab' in sys.modules
)

EMBEDDING_MODEL_NAME = "dinov2_vitb14"
EMBEDDING_MODEL_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EMBEDDING_BATCH_SIZE = 64


def _clean_secret(value: object, default: str | None = None) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else default
    return default if value is None else value


def _get_secret(getter, name: str, default: str | None = None) -> str | None:
    try:
        value = getter(name)
    except Exception:
        value = os.getenv(name)
    return _clean_secret(value, default)


if IS_KAGGLE or IS_COLAB:
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        _secret_getter = UserSecretsClient().get_secret
        KAGGLE_INPUT_DIR = Path("/kaggle/input")
        IMG_DATASET_NAME = "uav-detection-dataset"
        EXTRACTED_DATA_DIR = (
            KAGGLE_INPUT_DIR / "datasets" / "hannusiaknysh"
            / IMG_DATASET_NAME / "extracted_dataset"
        )
        WORKING_DIR = Path("/kaggle/working")
    else:
        from google.colab import userdata

        _secret_getter = userdata.get
        WORKING_DIR = Path("/content")
        EXTRACTED_DATA_DIR = Path(
            _get_secret(_secret_getter, "EXTRACTED_DATA_PATH")
            or str(WORKING_DIR / "extracted_dataset")
        )
        for _kaggle_key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            _secret_val = _get_secret(_secret_getter, _kaggle_key)
            if _secret_val and not os.getenv(_kaggle_key):
                os.environ[_kaggle_key] = _secret_val

    REPO_ID = _get_secret(_secret_getter, "REPO_ID")
    HF_TOKEN = _get_secret(_secret_getter, "HF_TOKEN")
    WANDB_API_KEY = _get_secret(_secret_getter, "WANDB_API_KEY")
    WANDB_PROJECT = _get_secret(_secret_getter, "WANDB_PROJECT", "drone-detection")

    FRAME_BASELINE_DIR = WORKING_DIR / "frame_baseline"
    OUTPUT_DIR = WORKING_DIR / "runs" / "detect"
    EMBEDDINGS_DIR = WORKING_DIR / "embeddings"
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
        "OUTPUT_DIR": Path("working") / "runs" / "detect",
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

    REPO_ID = _clean_secret(os.getenv("REPO_ID"))
    HF_TOKEN = _clean_secret(os.getenv("HF_TOKEN"))
    WANDB_API_KEY = _clean_secret(os.getenv("WANDB_API_KEY"))
    WANDB_PROJECT = _clean_secret(os.getenv("WANDB_PROJECT"), "drone-detection")

    HASH_THRESHOLD = 5

    CVAT_HOST = os.getenv("CVAT_HOST", "localhost")
    CVAT_PORT = os.getenv("CVAT_PORT", "8080")
    CVAT_USER = os.getenv("CVAT_USER")
    CVAT_PASS = os.getenv("CVAT_PASS")