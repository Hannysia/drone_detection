import copy
import csv
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.processing import manifests
from src.processing.manifests import read_manifest, write_manifest


@pytest.fixture
def row():
    return {
        "filename": "demo_f0001.jpg",
        "split": "train",
        "category": "positive",
    }


def test_read_baseline_adds_empty_exclusion_without_changing_file(tmp_path):
    path = tmp_path / "main.csv"
    original = (
        b"filename,split,category\r\n"
        b"demo_f0001.jpg,train,positive\r\n"
    )
    path.write_bytes(original)

    assert read_manifest(path) == [{
        "filename": "demo_f0001.jpg",
        "split": "train",
        "category": "positive",
        "excluded_at": "",
    }]
    assert path.read_bytes() == original


def test_round_trip_preserves_decisions_and_inputs(tmp_path, row):
    rows = [
        {
            **row,
            "filename": f"demo_f{index:04d}.jpg",
            "excluded_at": stage,
        }
        for index, stage in enumerate(
            ("", "iou", "embeddings"),
            1,
        )
    ]
    original = copy.deepcopy(rows)
    output = tmp_path / "selection.csv"

    assert write_manifest(rows, output) == 3
    assert read_manifest(output) == original
    assert rows == original


def test_deterministic_output_and_baseline_schema(tmp_path, row):
    rows = [
        {**row, "filename": "z.jpg", "split": "test"},
        {**row, "filename": "b.jpg"},
        {**row, "filename": "a.jpg"},
    ]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    write_manifest(rows, first, include_excluded=False)
    write_manifest(
        list(reversed(rows)),
        second,
        include_excluded=False,
    )

    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()

    with first.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        assert reader.fieldnames == ["filename", "split", "category"]
        assert [item["filename"] for item in reader] == [
            "a.jpg",
            "b.jpg",
            "z.jpg",
        ]


def test_existing_output_is_preserved(tmp_path, row):
    output = tmp_path / "main.csv"
    output.write_bytes(b"previous result")

    with pytest.raises(FileExistsError):
        write_manifest([row], output)

    assert output.read_bytes() == b"previous result"


def test_exclusion_cannot_be_dropped(tmp_path, row):
    output = tmp_path / "main.csv"

    with pytest.raises(ValueError, match="discard exclusion"):
        write_manifest(
            [{**row, "excluded_at": "iou"}],
            output,
            include_excluded=False,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "content",
    [
        "",
        "filename,split,category\n",
        "filename,split,category,category\na.jpg,train,positive,negative\n",
        "filename,split\na.jpg,train\n",
        "filename,split,category,unknown\na.jpg,train,positive,x\n",
        "filename,split,category\na.jpg,train\n",
        "filename,split,category\na.jpg,train,positive,extra\n",
        "filename,split,category\na.jpg,other,positive\n",
        "filename,split,category\na.jpg,train,other\n",
        "filename,split,category,excluded_at\na.jpg,train,positive,unknown\n",
        "filename,split,category\n../a.jpg,train,positive\n",
        "filename,split,category\nfolder\\a.jpg,train,positive\n",
        "filename,split,category\na.jpg,train,positive\na.jpg,train,negative\n",
    ],
)
def test_invalid_csv_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        read_manifest(path)


def test_same_filename_in_different_splits_is_allowed(tmp_path, row):
    path = tmp_path / "main.csv"
    write_manifest([row, {**row, "split": "val"}], path)

    assert len(read_manifest(path)) == 2


def test_bom_and_reordered_columns_are_supported(tmp_path):
    path = tmp_path / "main.csv"
    path.write_text(
        "category,filename,split\nnegative,a.jpg,val\n",
        encoding="utf-8-sig",
    )

    assert read_manifest(path) == [{
        "filename": "a.jpg",
        "split": "val",
        "category": "negative",
        "excluded_at": "",
    }]


def test_invalid_rows_do_not_create_output(tmp_path, row):
    output = tmp_path / "output" / "main.csv"

    with pytest.raises(ValueError, match="Duplicate"):
        write_manifest(
            [row, {**row, "category": "negative"}],
            output,
        )

    assert not output.parent.exists()


def test_failed_write_leaves_no_partial_result_and_can_retry(
    tmp_path, row, monkeypatch,
):
    output = tmp_path / "main.csv"

    def fail_write(writer, rows):
        writer.writerow(next(iter(rows)))
        raise OSError("injected disk failure")

    with monkeypatch.context() as changes:
        changes.setattr(csv.DictWriter, "writerows", fail_write)

        with pytest.raises(OSError, match="injected disk failure"):
            write_manifest([row], output)

    assert list(tmp_path.iterdir()) == []
    assert write_manifest([row], output) == 1


def test_two_writers_publish_one_complete_result(
    tmp_path, row, monkeypatch,
):
    output = tmp_path / "main.csv"
    barrier = Barrier(2)
    original_link = manifests.os.link

    def synchronized_link(source, destination):
        barrier.wait(timeout=10)
        original_link(source, destination)

    monkeypatch.setattr(manifests.os, "link", synchronized_link)
    candidates = [
        [row],
        [{**row, "filename": "other.jpg"}],
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_manifest, rows, output)
            for rows in candidates
        ]
        successful = 0
        rejected = 0

        for future in futures:
            try:
                assert future.result() == 1
                successful += 1
            except FileExistsError:
                rejected += 1

    assert (successful, rejected) == (1, 1)
    assert read_manifest(output)[0]["filename"] in {
        "demo_f0001.jpg",
        "other.jpg",
    }
    assert list(tmp_path.iterdir()) == [output]