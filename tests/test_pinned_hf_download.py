
import zipfile

from src.processing import extractor


def test_download_uses_pinned_revision(tmp_path, monkeypatch):
    revision = "a" * 40
    calls = []

    video = tmp_path / "video.mp4"
    video.write_bytes(b"test video")

    archive = tmp_path / "annotations.zip"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("gt/gt.txt", "1,1,10,20,30,40\n")

    def fake_download(**kwargs):
        calls.append(kwargs)

        if kwargs["filename"].endswith(".zip"):
            return str(archive)

        return str(video)

    monkeypatch.setattr(
        extractor,
        "hf_hub_download",
        fake_download,
    )

    _, gt_path = extractor.download_and_extract_assets(
        repo_id="test/dataset",
        hf_token="fake-token",
        video_path_hf="train/video.mp4",
        zip_path_hf="labels/train/batch_001/video.zip",
        tmp_dir=tmp_path,
        video_base_name="video",
        revision=revision,
    )

    assert len(calls) == 2
    assert all(
        call["revision"] == revision
        for call in calls
    )

    assert gt_path.read_text() == "1,1,10,20,30,40\n"
