
import pytest

from src.integration import cvat


BATCH_FILES = [
    "labels/test/batch_004/wing_thermal_sky_540.zip",
    "test/wing_thermal_sky_540.mp4",
]

CURATED_ZIP = (
    "labels_curated/v1/test/batch_004/"
    "wing_thermal_sky_540.zip"
)

CURATED_REVIEW = (
    "labels_curated/v1/test/batch_004/"
    "wing_thermal_sky_540.review.json"
)


def test_pull_skips_completed_video_on_hf(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: BATCH_FILES + [
            CURATED_ZIP,
            CURATED_REVIEW,
        ],
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Completed video must not be imported again"
        )

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        forbidden,
    )
    monkeypatch.setattr(
        cvat,
        "hf_hub_download",
        forbidden,
    )
    monkeypatch.setattr(
        cvat,
        "run_cmd",
        forbidden,
    )

    cvat.pull_batch_to_cvat(
        "test",
        "batch_004",
        annotation_source="labels",
        version="v1",
    )


@pytest.mark.parametrize(
    "existing_path",
    [CURATED_ZIP, CURATED_REVIEW],
)
def test_pull_rejects_incomplete_hf_result(
    monkeypatch,
    tmp_path,
    existing_path,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: BATCH_FILES + [existing_path],
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Incomplete HF result must block import"
        )

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        forbidden,
    )
    monkeypatch.setattr(
        cvat,
        "hf_hub_download",
        forbidden,
    )
    monkeypatch.setattr(
        cvat,
        "run_cmd",
        forbidden,
    )

    with pytest.raises(
        RuntimeError,
        match="Incomplete curated result",
    ):
        cvat.pull_batch_to_cvat(
            "test",
            "batch_004",
            annotation_source="labels",
            version="v1",
        )


def test_v1_completion_does_not_block_v2(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: BATCH_FILES + [
            CURATED_ZIP,
            CURATED_REVIEW,
        ],
    )

    checked_tasks = []

    def fake_find(task_name):
        checked_tasks.append(task_name)
        return [695]

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        fake_find,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Existing CVAT task must not be recreated"
        )

    monkeypatch.setattr(
        cvat,
        "hf_hub_download",
        forbidden,
    )
    monkeypatch.setattr(
        cvat,
        "run_cmd",
        forbidden,
    )

    cvat.pull_batch_to_cvat(
        "test",
        "batch_004",
        annotation_source="labels",
        version="v2",
    )

    assert checked_tasks == [
        "wing_thermal_sky_540_test"
    ]
