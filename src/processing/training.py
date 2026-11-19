import logging
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import mkdtemp

import torch
from huggingface_hub import HfApi
from ultralytics import YOLO

from src.config import (
    HF_TOKEN,
    OUTPUT_DIR,
    REPO_ID,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WORKING_DIR,
)
from src.processing.yolo_dataset import (
    _validate_run_name,
    cleanup_yolo_dataset,
    prepare_yolo_dataset,
)
from src.utils.visualization import plot_yolo_training_curves

log = logging.getLogger(__name__)


def _require_hf_credentials(
    repo_id: str | None = None,
    token: str | None = None,
) -> tuple[str, str]:
    target_repo = REPO_ID if repo_id is None else repo_id
    target_token = HF_TOKEN if token is None else token
    if not isinstance(target_repo, str) or not target_repo.strip():
        raise ValueError("REPO_ID is required for Hugging Face operations")
    if not isinstance(target_token, str) or not target_token.strip():
        raise ValueError("HF_TOKEN is required for Hugging Face operations")
    return target_repo.strip(), target_token.strip()


def assert_hf_ready(
    repo_id: str | None = None,
    token: str | None = None,
) -> tuple[str, str]:
    """Validates Hugging Face credentials, token write role, and repository access."""
    target_repo, target_token = _require_hf_credentials(repo_id, token)
    api = HfApi(token=target_token)
    identity = api.whoami()
    if isinstance(identity, dict):
        token_info = (identity.get("auth") or {}).get("accessToken") or {}
        if isinstance(token_info, dict) and token_info.get("role") == "read":
            raise PermissionError("HF_TOKEN has read-only access; a write token is required")
    api.repo_info(repo_id=target_repo, repo_type="dataset")
    return target_repo, target_token


def upload_checkpoint_to_hf(
    weights_path: Path,
    run_name: str,
    model_type: str = "yolo26n.pt",
) -> None:
    """Uploads best model weights to Hugging Face under the canonical models path."""
    validated_name = _validate_run_name(run_name)
    model_prefix = _validate_run_name(Path(model_type).stem)
    repo_id, token = _require_hf_credentials()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    log.info("Uploading checkpoint for %s to Hugging Face...", validated_name)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=f"models/{model_prefix}_{validated_name}_best.pt",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {model_prefix} checkpoint for {validated_name}",
    )
    log.info("Uploaded checkpoint for %s to Hugging Face.", validated_name)

def upload_run_artifacts_to_hf(run_dir: Path, run_name: str) -> None:
    """Uploads full YOLO training directory (weights, CSV, YAML, plots) to Hugging Face."""
    validated_name = _validate_run_name(run_name)
    repo_id, token = _require_hf_credentials()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run directory not found: {run_dir}")

    log.info("Uploading run artifacts for %s to Hugging Face...", validated_name)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=f"runs/{validated_name}",
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["*.cache", "__pycache__/*"],
        commit_message=f"Upload YOLO training artifacts for {validated_name}",
    )
    log.info("Uploaded run artifacts for %s to Hugging Face.", validated_name)


def _snapshot_run_files(save_dir: Path, snapshot_dir: Path) -> bool:
    copied = False
    for relative in (
        Path("weights") / "last.pt",
        Path("weights") / "best.pt",
        Path("results.csv"),
        Path("args.yaml"),
    ):
        source = save_dir / relative
        if source.is_file():
            target = snapshot_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied = True
    return copied
def _register_hf_backup_callback(
    model: YOLO,
    executor: ThreadPoolExecutor,
    run_name: str,
    repo_id: str,
    token: str,
    every_n_epochs: int,
) -> None:
    pending: Future | None = None
    def _upload_snapshot(snapshot_dir: Path, epoch_num: int) -> None:
        try:
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(snapshot_dir),
                path_in_repo=f"runs/{run_name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Backup YOLO {run_name} at epoch {epoch_num}",
            )
            log.info("Backed up %s snapshot (epoch %d) to Hugging Face.", run_name, epoch_num)
        except Exception as exc:
            log.warning(
                "Periodic HF backup failed at epoch %d for %s: %s",
                epoch_num,
                run_name,
                exc,
            )
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
    def _on_train_epoch_end(trainer) -> None:
        nonlocal pending
        epoch_num = int(getattr(trainer, "epoch", -1)) + 1
        total_epochs = getattr(trainer, "epochs", None)
        if epoch_num <= 0 or epoch_num % every_n_epochs != 0:
            return
        if (
            not isinstance(total_epochs, bool)
            and isinstance(total_epochs, int)
            and epoch_num >= total_epochs
        ):
            return
        if pending is not None and not pending.done():
            return
        raw_dir = getattr(trainer, "save_dir", None)
        if not isinstance(raw_dir, (str, Path)):
            return
        snapshot_dir = Path(mkdtemp(prefix=f"hf_backup_{run_name}_"))
        try:
            if not _snapshot_run_files(Path(raw_dir), snapshot_dir):
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                return
            pending = executor.submit(_upload_snapshot, snapshot_dir, epoch_num)
        except Exception:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)


def _resolve_wandb_key(api_key: str | None = None) -> str | None:
    candidate = WANDB_API_KEY if api_key is None else api_key
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    env_value = os.getenv("WANDB_API_KEY")
    return env_value.strip() if env_value and env_value.strip() else None


def _configure_ultralytics_wandb(enabled: bool) -> None:
    from ultralytics import settings

    if settings.get("wandb") != enabled:
        settings.update({"wandb": enabled})


def assert_wandb_ready(
    api_key: str | None = None,
    project: str | None = None,
) -> tuple[object, str, str]:
    """Validates Weights & Biases credentials, project name, and client authentication."""
    resolved_key = _resolve_wandb_key(api_key)
    if not resolved_key:
        raise ValueError("WANDB_API_KEY is required when Weights & Biases is enabled")

    target_project = WANDB_PROJECT if project is None else project
    if not isinstance(target_project, str) or not target_project.strip():
        raise ValueError("WANDB_PROJECT is required when Weights & Biases is enabled")

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb package is required when Weights & Biases is enabled") from exc

    if not wandb.login(key=resolved_key, relogin=True):
        raise RuntimeError("Weights & Biases login failed")

    return wandb, target_project.strip(), resolved_key


def _start_wandb_run(
    run_name: str,
    config: dict,
    use_wandb: bool | None,
) -> object | None:
    enabled = _resolve_wandb_key() is not None if use_wandb is None else use_wandb
    if not enabled:
        _configure_ultralytics_wandb(False)
        return None

    wandb, target_project, resolved_key = assert_wandb_ready()
    os.environ["WANDB_API_KEY"] = resolved_key
    os.environ["WANDB_PROJECT"] = target_project
    _configure_ultralytics_wandb(True)
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
    wandb.init(
        project=target_project,
        name=run_name,
        job_type="train",
        config=config,
        reinit=True,
    )
    return wandb


def _assert_gpu_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training; enable GPU accelerator before continuing"
        )


def run_yolo_training(
    yaml_path: Path,
    run_name: str,
    model_type: str = "yolo26n.pt",
    *,
    upload_to_hf: bool = True,
    use_wandb: bool | None = None,
    backup_every_n_epochs: int = 10,
    require_gpu: bool = False,
    **train_kwargs,
) -> Path:
    """Runs YOLO training with pre-flight checks, monitoring curves, W&B logging, and HF upload."""
    validated_name = _validate_run_name(run_name)
    if require_gpu:
        _assert_gpu_available()
    if backup_every_n_epochs < 0:
        raise ValueError("backup_every_n_epochs must be non-negative")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    hf_creds = assert_hf_ready() if upload_to_hf else None

    default_args = {
        "epochs": 50,
        "imgsz": 1280,
        "batch": 8,
        "patience": 15,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "optimizer": "auto",
        "rect": False,
        "cache": False,
        "workers": 2,
        "seed": 42,
        "deterministic": True,
        "project": str(OUTPUT_DIR),
        "name": validated_name,
        "exist_ok": True,
        "save": True,
        "plots": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "translate": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
    }
    default_args.update(train_kwargs)

    wb = _start_wandb_run(
        validated_name,
        {**default_args, "data": str(yaml_path), "model": model_type},
        use_wandb,
    )

    executor = (
        ThreadPoolExecutor(max_workers=1)
        if hf_creds is not None and backup_every_n_epochs > 0
        else None
    )
    failed = False
    try:
        log.info("Starting YOLO training run: %s (model=%s)", validated_name, model_type)
        model = YOLO(model_type)
        if executor is not None and isinstance(hf_creds, tuple) and len(hf_creds) == 2:
            _register_hf_backup_callback(
                model,
                executor,
                validated_name,
                hf_creds[0],
                hf_creds[1],
                backup_every_n_epochs,
            )
        results = model.train(data=str(yaml_path), **default_args)

        best_weights = Path(model.trainer.best)
        if not best_weights.is_file():
            raise FileNotFoundError(f"Checkpoint missing after training: {best_weights}")

        raw_save_dir = getattr(model.trainer, "save_dir", None)
        run_dir = (
            Path(raw_save_dir)
            if isinstance(raw_save_dir, (str, Path))
            else best_weights.parent.parent
        )

        if results is not None and hasattr(results, "results_dict"):
            metrics = results.results_dict
            log.info(
                "Metrics (%s): P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f",
                validated_name,
                metrics.get("metrics/precision(B)", 0.0),
                metrics.get("metrics/recall(B)", 0.0),
                metrics.get("metrics/mAP50(B)", 0.0),
                metrics.get("metrics/mAP50-95(B)", 0.0),
            )

        results_csv = run_dir / "results.csv"
        if results_csv.is_file():
            curves_path = run_dir / f"training_curves_{validated_name}.png"
            summary = plot_yolo_training_curves(
                results_csv,
                validated_name,
                save_path=curves_path,
                show=False,
            )
            log.info(
                "Training monitoring (%s): best_epoch=%s/%s mAP50=%s mAP50-95=%s gap=%s status=%s",
                validated_name,
                summary["best_epoch"],
                summary["last_epoch"],
                summary["best_map50"],
                summary["best_map50_95"],
                summary["val_train_loss_gap"],
                summary["fit_status"],
            )
            if wb is not None and getattr(wb, "run", None) is not None:
                if hasattr(wb, "log") and hasattr(wb, "Image"):
                    wb.log({"monitoring/training_curves": wb.Image(str(curves_path))})
                run_summary = getattr(wb.run, "summary", None)
                if run_summary is not None and hasattr(run_summary, "update"):
                    run_summary.update({
                        k: v for k, v in summary.items() if v is not None
                    })
    except Exception:
        failed = True
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if wb is not None and getattr(wb, "run", None) is not None:
            wb.finish(exit_code=1 if failed else 0)

    if upload_to_hf:
        upload_checkpoint_to_hf(best_weights, validated_name, model_type=model_type)
        upload_run_artifacts_to_hf(run_dir, validated_name)

    return best_weights


def run_full_pipeline(
    use_negatives: bool,
    run_name: str,
    *,
    dataset_root: Path,
    source_path: Path | None = None,
    overwrite: bool = True,
    model_type: str = "yolo26n.pt",
    upload_to_hf: bool = True,
    use_wandb: bool | None = None,
    backup_every_n_epochs: int = 10,
    cleanup_dataset: bool = True,
    require_gpu: bool = False,
    **train_kwargs,
) -> Path:
    """Prepares dataset, runs YOLO training with monitoring/uploads, and cleans staged data."""
    validated_name = _validate_run_name(run_name)
    if require_gpu:
        _assert_gpu_available()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    if upload_to_hf:
        assert_hf_ready()

    enabled_wandb = _resolve_wandb_key() is not None if use_wandb is None else use_wandb
    if enabled_wandb:
        assert_wandb_ready()

    yaml_path = prepare_yolo_dataset(
        dataset_root,
        WORKING_DIR,
        validated_name,
        use_negatives,
        source_path=source_path,
        overwrite=overwrite,
    )
    try:
        return run_yolo_training(
            yaml_path,
            validated_name,
            model_type=model_type,
            upload_to_hf=upload_to_hf,
            use_wandb=use_wandb,
            backup_every_n_epochs=backup_every_n_epochs,
            **train_kwargs,
        )
    finally:
        if cleanup_dataset:
            cleanup_yolo_dataset(WORKING_DIR, validated_name)