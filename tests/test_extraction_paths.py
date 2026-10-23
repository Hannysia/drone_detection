
import pytest

from src.utils.io import get_extraction_paths


def test_extraction_paths_are_isolated(tmp_path):
    historical = tmp_path / "extracted_dataset"
    historical.mkdir()

    original = historical / "keep.txt"
    original.write_text("historical", encoding="utf-8")

    baseline = tmp_path / "frame_baseline_v1"

    output, temporary, log_file = get_extraction_paths(
        baseline,
        historical,
    )

    assert output == baseline
    assert temporary == baseline / ".tmp"
    assert log_file == baseline / ".state" / "processed_videos.log"

    assert original.read_text(encoding="utf-8") == "historical"
    assert not baseline.exists()


@pytest.mark.parametrize(
    "bad_output",
    ["same", "nested", "parent"],
)
def test_rejects_overlapping_paths(tmp_path, bad_output):
    historical = tmp_path / "extracted_dataset"
    historical.mkdir()

    outputs = {
        "same": historical,
        "nested": historical / "frame_baseline_v1",
        "parent": tmp_path,
    }

    with pytest.raises(ValueError, match="must not overlap"):
        get_extraction_paths(
            outputs[bad_output],
            historical,
        )
