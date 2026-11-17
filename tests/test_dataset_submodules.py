from src.processing import dataset_maintenance


def test_dataset_maintenance_unify_labels_standalone(tmp_path):
    pos_dir = tmp_path / "train" / "RAW_POSITIVES"
    pos_dir.mkdir(parents=True)
    label = pos_dir / "wing_rgb_sky_001_f0001.txt"
    label.write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    dataset_maintenance.unify_labels_logic(tmp_path, ["train"], "0")

    assert label.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"