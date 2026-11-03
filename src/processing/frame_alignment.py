def build_frame_mapping(review: dict) -> dict[int, int]:
    required = {
        "size",
        "start_frame",
        "stop_frame",
        "frame_filter",
        "included_frames",
        "deleted_frames",
    }
    missing = required - review.keys()

    if missing:
        raise ValueError(f"Missing review fields: {sorted(missing)}")

    size = review["size"]

    if type(size) is not int or size <= 0:
        raise ValueError("Invalid review size")

    if (
        type(review["start_frame"]) is not int
        or type(review["stop_frame"]) is not int
        or review["start_frame"] != 0
        or review["stop_frame"] != size - 1
        or review["frame_filter"] != ""
        or review["included_frames"] is not None
    ):
        raise ValueError("Only full videos without frame sampling are supported")

    deleted = review["deleted_frames"]

    if (
        not isinstance(deleted, list)
        or any(type(index) is not int or not 0 <= index < size for index in deleted)
        or len(deleted) != len(set(deleted))
    ):
        raise ValueError("Invalid deleted_frames")

    deleted = set(deleted)
    retained = (index + 1 for index in range(size) if index not in deleted)

    return {source: mot for mot, source in enumerate(retained, start=1)}