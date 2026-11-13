def test_seqmap_lists_only_successful_sequences(tmp_path, monkeypatch):
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    monkeypatch.setenv("LABELS_DIR", str(labels))
    monkeypatch.setenv("TRACKEVAL_DIR", str(output))

    from scripts import prepare_trackeval as script

    monkeypatch.setattr(script, "LABELS_DIR", labels)
    monkeypatch.setattr(script, "TRACKEVAL_DIR", output)

    split = labels / "val"
    split.mkdir(parents=True)
    for name in ("failed", "ready"):
        (split / f"{name}.mp4").touch()
        (labels / f"{name}.zip").touch()

    attempted = set()

    def convert(*, video_path, **kwargs):
        attempted.add(video_path.stem)
        if video_path.stem == "failed":
            raise ValueError("invalid archive")

    monkeypatch.setattr(script, "prepare_trackeval_sequence", convert)
    script.main()

    seqmap = output / "seqmaps" / "drone-val.txt"
    assert attempted == {"failed", "ready"}
    assert seqmap.read_text(encoding="utf-8").splitlines() == ["name", "ready"]