import importlib
import json

import pytest


def test_short_workspace_resume_and_revision_change(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.05_extract_data")
    source = {
        "repo_id": "test/dataset",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "annotation_version": "v1",
    }
    source_path = tmp_path / "manifests" / "sources" / "hf.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(source))
    frames = tmp_path / "frames"
    workspace = frames / "v1"
    archive = "labels_curated/v1/train/batch_001/demo.zip"
    network_calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def list_repo_files(self, **kwargs):
            assert json.loads((workspace / "source.json").read_text()) == source
            assert kwargs["revision"] == source["revision"]
            network_calls.append(kwargs)
            return [
                "train/demo.mp4",
                archive,
                archive.removesuffix(".zip") + ".review.json",
            ]

    downloads = []

    def fake_download(*args, **kwargs):
        downloads.append(kwargs)
        video = tmp_path / "download.mp4"
        video.write_bytes(b"video")
        gt = tmp_path / "gt.txt"
        gt.write_text("1,1,0,0,8,8\n")
        return video, gt, {"video_name": "demo"}

    def fake_extract(video, annotations, name, positive, negative, *, review):
        assert positive == workspace / "train" / "RAW_POSITIVES"
        assert review["video_name"] == name == "demo"
        (positive / "demo_f0001.jpg").write_bytes(b"image")
        (positive / "demo_f0001.txt").write_text("0 0.5 0.5 0.2 0.2")

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "FRAME_BASELINE_DIR", frames)
    monkeypatch.setattr(script, "EXTRACTED_DATA_DIR", tmp_path / "historical")
    monkeypatch.setattr(script, "REPO_ID", source["repo_id"])
    monkeypatch.setattr(script, "SPLITS", ["train"])
    monkeypatch.setattr(script, "HfApi", FakeApi)
    monkeypatch.setattr(script, "download_and_extract_assets", fake_download)
    monkeypatch.setattr(script, "extract_frames_and_labels", fake_extract)

    script.main("train", "demo")
    script.main("train", "demo")

    assert len(downloads) == 1
    assert len(network_calls) == 2
    assert not (workspace / source["revision"]).exists()
    assert (workspace / ".state" / "processed_videos.log").read_text() == "demo\n"

    source_path.write_text(json.dumps({**source, "revision": "b" * 40}))

    with pytest.raises(ValueError, match="source mismatch"):
        script.main("train", "demo")

    assert len(network_calls) == 2
    assert len(downloads) == 1