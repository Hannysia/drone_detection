import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_training(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXTRACTED_DATA_PATH", str(tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    from src.processing import training

    monkeypatch.setattr(training, "WANDB_API_KEY", None)
    monkeypatch.setattr(training, "_configure_ultralytics_wandb", Mock())
    return training


def test_pipeline_requires_explicit_dataset_root_and_cleans_up(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    root = tmp_path / "curated"
    root.mkdir()
    yaml_path = tmp_path / "dataset.yaml"
    weights = tmp_path / "best.pt"

    hf_check = Mock()
    prepare = Mock(return_value=yaml_path)
    fit = Mock(return_value=weights)
    cleanup = Mock()
    evaluate = Mock()

    monkeypatch.setattr(training, "assert_hf_ready", hf_check)
    monkeypatch.setattr(training, "prepare_yolo_dataset", prepare)
    monkeypatch.setattr(training, "run_yolo_training", fit)
    monkeypatch.setattr(training, "cleanup_yolo_dataset", cleanup)
    monkeypatch.setitem(
        sys.modules,
        "src.processing.evaluation",
        SimpleNamespace(evaluate_model=evaluate),
    )

    with pytest.raises(TypeError):
        training.run_full_pipeline(False, "example")

    assert training.run_full_pipeline(False, "example", dataset_root=root, epochs=25) == weights
    hf_check.assert_called_once_with()
    prepare.assert_called_once_with(
        root,
        training.WORKING_DIR,
        "example",
        False,
        source_path=None,
        overwrite=True,
    )
    fit.assert_called_once_with(
        yaml_path,
        "example",
        model_type="yolo26n.pt",
        upload_to_hf=True,
        use_wandb=None,
        backup_every_n_epochs=10,
        epochs=25,
    )
    cleanup.assert_called_once_with(training.WORKING_DIR, "example")
    evaluate.assert_not_called()


def test_pipeline_cleans_up_even_when_training_fails(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    root = tmp_path / "curated"
    root.mkdir()
    cleanup = Mock()

    monkeypatch.setattr(training, "prepare_yolo_dataset", Mock(return_value=tmp_path / "d.yaml"))
    monkeypatch.setattr(training, "run_yolo_training", Mock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(training, "cleanup_yolo_dataset", cleanup)

    with pytest.raises(RuntimeError, match="boom"):
        training.run_full_pipeline(True, "run", dataset_root=root, upload_to_hf=False)

    cleanup.assert_called_once_with(training.WORKING_DIR, "run")


def test_training_uses_safe_defaults_plots_curves_and_uploads_artifacts(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    run_dir = tmp_path / "run10"
    current = run_dir / "weights" / "best.pt"
    current.parent.mkdir(parents=True)
    current.touch()

    results_csv = run_dir / "results.csv"
    results_csv.write_text(
        "epoch,train/box_loss,train/cls_loss,val/box_loss,val/cls_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,1.0,0.8,1.1,0.9,0.80,0.75,0.78,0.52\n",
        encoding="utf-8",
    )

    model = Mock()
    model.train.return_value = None
    model.trainer.best = current
    model.trainer.save_dir = run_dir

    hf_check = Mock()
    upload_ckpt = Mock()
    upload_dir = Mock()
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(training, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(training, "assert_hf_ready", hf_check)
    monkeypatch.setattr(training, "upload_checkpoint_to_hf", upload_ckpt)
    monkeypatch.setattr(training, "upload_run_artifacts_to_hf", upload_dir)

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")

    assert training.run_yolo_training(yaml_path, "run") == current
    hf_check.assert_called_once_with()
    assert model.train.call_args.kwargs["cache"] is False
    assert model.train.call_args.kwargs["rect"] is False
    assert model.train.call_args.kwargs["plots"] is False
    assert model.train.call_args.kwargs["imgsz"] == 1280
    assert model.train.call_args.kwargs["batch"] == 8
    assert model.train.call_args.kwargs["epochs"] == 50
    assert model.train.call_args.kwargs["patience"] == 15
    assert (run_dir / "training_curves_run.png").is_file()
    upload_ckpt.assert_called_once_with(current, "run", model_type="yolo26n.pt")
    upload_dir.assert_called_once_with(run_dir, "run")

    current.unlink()
    with pytest.raises(FileNotFoundError):
        training.run_yolo_training(yaml_path, "run")
    assert upload_ckpt.call_count == 1
    assert upload_dir.call_count == 1


def test_training_fails_fast_before_yolo_when_hf_or_wandb_unready(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    yolo_cls = Mock()
    monkeypatch.setattr(training, "YOLO", yolo_cls)
    monkeypatch.setattr(training, "REPO_ID", None)
    monkeypatch.setattr(training, "HF_TOKEN", None)

    with pytest.raises(ValueError, match="REPO_ID"):
        training.run_yolo_training(yaml_path, "run", upload_to_hf=True)

    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        training.run_yolo_training(yaml_path, "run", upload_to_hf=False, use_wandb=True)

    yolo_cls.assert_not_called()


def test_assert_hf_ready_validates_credentials_role_and_repo(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    monkeypatch.setattr(training, "REPO_ID", " ")
    monkeypatch.setattr(training, "HF_TOKEN", "tok")
    with pytest.raises(ValueError, match="REPO_ID"):
        training.assert_hf_ready()

    api_read = Mock()
    api_read.whoami.return_value = {"auth": {"accessToken": {"role": "read"}}}
    monkeypatch.setattr(training, "REPO_ID", "user/repo")
    monkeypatch.setattr(training, "HF_TOKEN", "hf_read")
    monkeypatch.setattr(training, "HfApi", Mock(return_value=api_read))
    with pytest.raises(PermissionError, match="read-only"):
        training.assert_hf_ready()

    api_write = Mock()
    api_write.whoami.return_value = {"auth": {"accessToken": {"role": "write"}}}
    monkeypatch.setattr(training, "HfApi", Mock(return_value=api_write))
    assert training.assert_hf_ready() == ("user/repo", "hf_read")
    api_write.repo_info.assert_called_once_with(repo_id="user/repo", repo_type="dataset")


def test_upload_helpers_validate_and_propagate_errors(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    weights = tmp_path / "best.pt"
    run_dir = tmp_path / "run_dir"

    monkeypatch.setattr(training, "REPO_ID", None)
    monkeypatch.setattr(training, "HF_TOKEN", None)
    with pytest.raises(ValueError, match="REPO_ID"):
        training.upload_checkpoint_to_hf(weights, "run")
    with pytest.raises(ValueError, match="REPO_ID"):
        training.upload_run_artifacts_to_hf(run_dir, "run")

    monkeypatch.setattr(training, "REPO_ID", "repo/test")
    monkeypatch.setattr(training, "HF_TOKEN", "token-test")
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        training.upload_checkpoint_to_hf(weights, "run")
    with pytest.raises(FileNotFoundError, match="Training run directory not found"):
        training.upload_run_artifacts_to_hf(run_dir, "run")

    weights.write_bytes(b"pt")
    run_dir.mkdir()
    api = Mock()
    api.upload_file.side_effect = RuntimeError("hf upload_file error")
    api.upload_folder.side_effect = RuntimeError("hf upload_folder error")
    monkeypatch.setattr(training, "HfApi", Mock(return_value=api))

    with pytest.raises(RuntimeError, match="hf upload_file error"):
        training.upload_checkpoint_to_hf(weights, "run")
    with pytest.raises(RuntimeError, match="hf upload_folder error"):
        training.upload_run_artifacts_to_hf(run_dir, "run")

    api.upload_file.side_effect = None
    training.upload_checkpoint_to_hf(weights, "run", model_type="yolo26s.pt")
    assert api.upload_file.call_args.kwargs["path_in_repo"] == "models/yolo26s_run_best.pt"


def test_wandb_lifecycle_finishes_on_success_and_error(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    best = tmp_path / "run_wb" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.touch()

    fake_wandb = SimpleNamespace(
        run=object(),
        login=Mock(return_value=True),
        init=Mock(),
        finish=Mock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(training, "WANDB_API_KEY", "wb-secret")
    monkeypatch.setattr(training, "WANDB_PROJECT", "uav-proj")

    ok_model = Mock()
    ok_model.train.return_value = None
    ok_model.trainer.best = best
    monkeypatch.setattr(training, "YOLO", Mock(return_value=ok_model))

    training.run_yolo_training(yaml_path, "run_wb", upload_to_hf=False, use_wandb=True)
    fake_wandb.login.assert_called_once_with(key="wb-secret", relogin=True)
    assert fake_wandb.init.call_args.kwargs["project"] == "uav-proj"
    assert fake_wandb.init.call_args.kwargs["name"] == "run_wb"
    fake_wandb.finish.assert_any_call(exit_code=0)

    fake_wandb.finish.reset_mock()
    bad_model = Mock()
    bad_model.train.side_effect = RuntimeError("cuda oom")
    monkeypatch.setattr(training, "YOLO", Mock(return_value=bad_model))

    with pytest.raises(RuntimeError, match="cuda oom"):
        training.run_yolo_training(yaml_path, "run_wb", upload_to_hf=False, use_wandb=True)
    fake_wandb.finish.assert_any_call(exit_code=1)


class _InlineExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        done = Mock()
        done.done.return_value = True
        return done

    def shutdown(self, wait=True):
        pass


def test_periodic_hf_backup_snapshots_intervals_and_tolerates_errors(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")

    run_dir = tmp_path / "run_bk"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    (run_dir / "results.csv").write_text("epoch,train/box_loss\n1,1.0\n", encoding="utf-8")

    callbacks = {}
    uploaded_snapshots = []

    api = Mock()
    api.whoami.return_value = {"auth": {"accessToken": {"role": "write"}}}

    def fake_upload_folder(**kwargs):
        folder = training.Path(kwargs["folder_path"])
        files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        uploaded_snapshots.append((kwargs["path_in_repo"], kwargs["commit_message"], files))
        if "epoch 10" in kwargs["commit_message"]:
            raise RuntimeError("transient blip")

    api.upload_folder.side_effect = fake_upload_folder
    monkeypatch.setattr(training, "ThreadPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(training, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(training, "REPO_ID", "user/repo")
    monkeypatch.setattr(training, "HF_TOKEN", "hf_tok")
    monkeypatch.setattr(training, "upload_checkpoint_to_hf", Mock())
    monkeypatch.setattr(training, "upload_run_artifacts_to_hf", Mock())
    
    model = Mock()
    model.trainer.best = best
    model.trainer.save_dir = run_dir
    model.add_callback.side_effect = lambda event, fn: callbacks.__setitem__(event, fn)

    def fake_train(**kwargs):
        cb = callbacks["on_train_epoch_end"]
        cb(SimpleNamespace(epoch=1, epochs=30, save_dir=run_dir))
        cb(SimpleNamespace(epoch=9, epochs=30, save_dir=run_dir))
        cb(SimpleNamespace(epoch=19, epochs=30, save_dir=run_dir))
        cb(SimpleNamespace(epoch=29, epochs=30, save_dir=run_dir))

    model.train.side_effect = fake_train
    monkeypatch.setattr(training, "YOLO", Mock(return_value=model))

    result = training.run_yolo_training(
        yaml_path,
        "run_bk",
        upload_to_hf=True,
        backup_every_n_epochs=10,
    )

    assert result == best
    assert uploaded_snapshots == [
        ("runs/run_bk", "Backup YOLO run_bk at epoch 10", ["results.csv", "weights/best.pt", "weights/last.pt"]),
        ("runs/run_bk", "Backup YOLO run_bk at epoch 20", ["results.csv", "weights/best.pt", "weights/last.pt"]),
    ]


def test_require_gpu_fails_fast_when_cuda_unavailable(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)
    root = tmp_path / "curated"
    root.mkdir()
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    prepare = Mock()
    yolo_cls = Mock()
    monkeypatch.setattr(training, "prepare_yolo_dataset", prepare)
    monkeypatch.setattr(training, "YOLO", yolo_cls)
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        training.run_yolo_training(
            yaml_path,
            "run_gpu",
            upload_to_hf=False,
            require_gpu=True,
        )
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        training.run_full_pipeline(
            False,
            "run_gpu",
            dataset_root=root,
            upload_to_hf=False,
            require_gpu=True,
        )
    prepare.assert_not_called()
    yolo_cls.assert_not_called()
    fit = Mock(return_value=tmp_path / "best.pt")
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "prepare_yolo_dataset", Mock(return_value=yaml_path))
    monkeypatch.setattr(training, "run_yolo_training", fit)
    monkeypatch.setattr(training, "cleanup_yolo_dataset", Mock())
    assert (
        training.run_full_pipeline(
            False,
            "run_gpu",
            dataset_root=root,
            upload_to_hf=False,
            require_gpu=True,
        )
        == tmp_path / "best.pt"
    )
    fit.assert_called_once()


def test_wandb_logs_monitoring_curves_and_summary(tmp_path, monkeypatch):
    training = _load_training(tmp_path, monkeypatch)

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    run_dir = tmp_path / "run_wb_curves"
    best = run_dir / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.touch()
    (run_dir / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,val/box_loss,val/cls_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,1.0,0.8,1.1,0.9,0.80,0.75,0.78,0.52\n",
        encoding="utf-8",
    )

    fake_run = SimpleNamespace(summary={})
    fake_wandb = SimpleNamespace(
        run=fake_run,
        login=Mock(return_value=True),
        init=Mock(),
        log=Mock(),
        Image=lambda path: ("wandb_image", path),
        finish=Mock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(training, "WANDB_API_KEY", "wb-secret")
    monkeypatch.setattr(training, "WANDB_PROJECT", "uav-proj")

    model = Mock()
    model.train.return_value = None
    model.trainer.best = best
    model.trainer.save_dir = run_dir
    monkeypatch.setattr(training, "YOLO", Mock(return_value=model))

    training.run_yolo_training(yaml_path, "run_wb_curves", upload_to_hf=False, use_wandb=True)

    curves_path = run_dir / "training_curves_run_wb_curves.png"
    fake_wandb.log.assert_called_once_with(
        {"monitoring/training_curves": ("wandb_image", str(curves_path))}
    )
    assert fake_run.summary["best_epoch"] == 1
    assert "fit_status" in fake_run.summary