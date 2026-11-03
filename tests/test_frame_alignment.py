import pytest

from src.processing.frame_alignment import build_frame_mapping


def make_review(size=7, deleted=None):
    return {
        "size": size,
        "start_frame": 0,
        "stop_frame": size - 1,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [] if deleted is None else deleted,
    }


def test_no_deletions():
    assert build_frame_mapping(make_review(size=3)) == {1: 1, 2: 2, 3: 3}


def test_first_middle_and_last_frames_are_excluded():
    review = make_review(deleted=[6, 0, 3])

    assert build_frame_mapping(review) == {2: 1, 3: 2, 5: 3, 6: 4}
    assert review["deleted_frames"] == [6, 0, 3]


def test_unannotated_frames_keep_their_mot_positions():
    mapping = build_frame_mapping(make_review(size=6, deleted=[1]))
    annotations = {1: [[10, 20, 30, 40]], 4: [[50, 60, 70, 80]]}
    aligned = {source: annotations.get(mot, []) for source, mot in mapping.items()}

    assert aligned == {
        1: [[10, 20, 30, 40]],
        3: [],
        4: [],
        5: [[50, 60, 70, 80]],
        6: [],
    }


def test_sixty_nine_frame_example():
    review = make_review(size=69, deleted=[1, 4, 8, 33, 37, 68])
    mapping = build_frame_mapping(review)

    assert len(mapping) == 63
    assert mapping[1] == 1
    assert mapping[3] == 2
    assert mapping[10] == 7
    assert mapping[68] == 63
    assert {2, 5, 9, 34, 38, 69}.isdisjoint(mapping)
    assert list(mapping.values()) == list(range(1, 64))


def test_all_frames_deleted():
    assert build_frame_mapping(make_review(size=2, deleted=[0, 1])) == {}


@pytest.mark.parametrize("field", list(make_review()))
def test_missing_metadata_is_rejected(field):
    review = make_review()
    del review[field]

    with pytest.raises(ValueError, match="Missing review fields"):
        build_frame_mapping(review)


@pytest.mark.parametrize("size", [0, -1, True, 7.0, "7", None])
def test_invalid_size_is_rejected(size):
    review = make_review()
    review["size"] = size

    with pytest.raises(ValueError, match="Invalid review size"):
        build_frame_mapping(review)


@pytest.mark.parametrize(
    "deleted",
    [None, "1", [1, 1], [-1], [7], [True], [1.5], [[1]]],
)
def test_invalid_deleted_frames_are_rejected(deleted):
    review = make_review()
    review["deleted_frames"] = deleted

    with pytest.raises(ValueError, match="Invalid deleted_frames"):
        build_frame_mapping(review)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_frame", 1),
        ("start_frame", False),
        ("stop_frame", 5),
        ("stop_frame", 6.0),
        ("frame_filter", "step=2"),
        ("included_frames", [0, 2, 4, 6]),
    ],
)
def test_unsupported_frame_selection_is_rejected(field, value):
    review = make_review()
    review[field] = value

    with pytest.raises(ValueError, match="Only full videos"):
        build_frame_mapping(review)
        