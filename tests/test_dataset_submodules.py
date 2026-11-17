from src.processing import dataset, dataset_maintenance, trackeval_dataset, yolo_dataset


def test_dataset_facade_reexports_split_submodules():
    assert dataset.TRASH_FOLDER_NAMES is dataset_maintenance.TRASH_FOLDER_NAMES
    assert dataset.list_trash_folders_logic is dataset_maintenance.list_trash_folders_logic
    assert dataset.restore_from_trash_logic is dataset_maintenance.restore_from_trash_logic
    assert dataset.unify_labels_logic is dataset_maintenance.unify_labels_logic
    assert dataset.prepare_trackeval_sequence is trackeval_dataset.prepare_trackeval_sequence
    assert dataset.compile_kaggle_mot_dataset is trackeval_dataset.compile_kaggle_mot_dataset
    assert dataset.safe_symlink is yolo_dataset.safe_symlink
    assert dataset.prepare_yolo_dataset is yolo_dataset.prepare_yolo_dataset


def test_dataset_maintenance_unify_labels_standalone(tmp_path):
    pos_dir = tmp_path / "train" / "RAW_POSITIVES"
    pos_dir.mkdir(parents=True)
    label = pos_dir / "wing_rgb_sky_001_f0001.txt"
    label.write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    dataset_maintenance.unify_labels_logic(tmp_path, ["train"], "0")

    assert label.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"