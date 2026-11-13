import zipfile

import pytest

from src.processing import dataset


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "get_video_metadata",
        lambda _: (640, 480, 25, 3),
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()

    return (
        tmp_path / "example.mp4",
        tmp_path / "example.zip",
        gt_dir,
        tmp_path / "seqinfo.ini",
    )


@pytest.mark.parametrize(
    "contents",
    [b"", b" \r\n", b"1,1,2,3,4,5,1,2,1\n"],
)
def test_sequence_accepts_present_gt(sequence, contents):
    video, archive, folder, info = sequence

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("gt/gt.txt", contents)

    dataset.prepare_trackeval_sequence(video, archive, folder, info)

    expected = "1,1,2.00,3.00,4.00,5.00,1,1,1.00\n" if contents.strip() else ""
    assert (folder / "gt.txt").read_text() == expected
    assert "seqLength=3" in info.read_text()


def test_sequence_rejects_missing_gt(sequence):
    video, archive, folder, info = sequence

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("not_gt.txt", b"")

    with pytest.raises(ValueError, match="expected exactly one gt.txt, found 0"):
        dataset.prepare_trackeval_sequence(video, archive, folder, info)

    assert not (folder / "gt.txt").exists()
    assert not info.exists()