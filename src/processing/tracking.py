import configparser
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from src.processing.frame_alignment import build_frame_mapping

log = logging.getLogger(__name__)

from boxmot.trackers import ByteTrack, OcSort

logging.getLogger("boxmot").setLevel(logging.WARNING)
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except ImportError:
    pass

def get_configured_tracker(tracker_name: str, config: dict, fps: int):
    """Factory to initialize a tracker with dynamic configurations."""
    if tracker_name.lower() == "bytetrack":
        return ByteTrack(
            track_thresh=config.get("track_thresh", 0.5),
            match_thresh=config.get("match_thresh", 0.8),
            track_buffer=int(config.get("track_buffer", fps))
        )
    elif tracker_name.lower() == "ocsort":
        return OcSort(
            det_thresh=config.get("det_thresh", 0.5),
            max_age=int(config.get("max_age", fps)),
            iou_threshold=config.get("iou_threshold", 0.3)
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_name}")

def run_tracking_pipeline(
    split_name: str, 
    tracker_name: str, 
    tracker_config: dict, 
    run_dir_name: str, 
    dataset_base_dir: Path, 
    trackers_out_dir: Path,
    detections_source_dir: Path = None
) -> dict[str, float]:
    """Runs tracking and measures high-precision tracker execution time (latency and FPS)."""
    split_dir = dataset_base_dir / f"drone-{split_name}"
    if not split_dir.exists():
        log.error("Split directory %s does not exist.", split_dir)
        return {"avg_latency_ms": 0.0, "avg_fps": 0.0, "total_frames": 0, "total_time_s": 0.0}

    sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
    global_total_time = 0.0
    global_total_frames = 0

    for seq in sequences:
        seq_name = seq.name
        seqinfo_file = seq / "seqinfo.ini"
        if not seqinfo_file.is_file():
            raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_file}")

        seq_cfg = configparser.ConfigParser()
        seq_cfg.read(seqinfo_file, encoding="utf-8")
        width = int(seq_cfg["Sequence"]["imWidth"])
        height = int(seq_cfg["Sequence"]["imHeight"])
        fps = int(seq_cfg["Sequence"]["frameRate"])
        seq_length = int(seq_cfg["Sequence"]["seqLength"])
        if min(width, height, fps, seq_length) <= 0:
            raise ValueError(
                f"Invalid seqinfo.ini values in {seqinfo_file}: w={width}, h={height}, fps={fps}, len={seq_length}"
            )
        
        tracker = get_configured_tracker(tracker_name, tracker_config, fps)
        
        if detections_source_dir is None:
            det_file = seq / "gt/gt.txt"
        else:
            det_file = detections_source_dir / f"{seq_name}.txt"
            
        if not det_file.exists():
            continue
            
        if det_file.stat().st_size > 0:
            detections = np.loadtxt(det_file, delimiter=',')
            if detections.ndim == 1:
                detections = np.array([detections])
        else:
            detections = np.empty((0, 9))
            
        results = []
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        seq_tracker_time = 0.0
        seq_total_frames = 0
        
        for frame_id in range(1, seq_length + 1):
            if len(detections) > 0:
                frame_dets = detections[detections[:, 0] == frame_id]
            else:
                frame_dets = np.empty((0, 9))
                
            if frame_dets.ndim == 1:
                frame_dets = np.array([frame_dets])
                
            tracker_input = []
            if len(frame_dets) > 0:
                for det in frame_dets:
                    left, top, w, h = det[2:6]
                    conf = det[6] if (detections_source_dir is not None and det[6] > 0) else 1.0
                    cls = 1
                    tracker_input.append([left, top, left + w, top + h, conf, cls])
                
            tracker_input = np.array(tracker_input) if len(tracker_input) > 0 else np.empty((0, 6))
            
            start_time = time.perf_counter()
            online_targets = tracker.update(tracker_input, dummy_frame)
            end_time = time.perf_counter()
            
            seq_tracker_time += (end_time - start_time)
            seq_total_frames += 1
            
            for target in online_targets:
                x1, y1, x2, y2, track_id, conf, cls = target[:7]
                results.append([frame_id, track_id, x1, y1, x2 - x1, y2 - y1, conf, -1, -1, -1])
                
        global_total_time += seq_tracker_time
        global_total_frames += seq_total_frames
        
        out_path = trackers_out_dir / f"drone-{split_name}" / run_dir_name / "data"
        out_path.mkdir(parents=True, exist_ok=True)
        
        res_file = out_path / f"{seq_name}.txt"
        if len(results) > 0:
            np.savetxt(res_file, np.array(results), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        else:
            res_file.write_text("", encoding="utf-8")

    if global_total_frames > 0 and global_total_time > 0:
        global_latency_ms = (global_total_time / global_total_frames) * 1000
        global_fps = global_total_frames / global_total_time
        log.info(
            "===> [%s] %s | Config: %s | Avg Latency: %.3f ms/frame | Avg FPS: %.1f",
            tracker_name.upper(),
            run_dir_name.upper(),
            tracker_config,
            global_latency_ms,
            global_fps,
        )
        return {
            "avg_latency_ms": float(global_latency_ms),
            "avg_fps": float(global_fps),
            "total_frames": int(global_total_frames),
            "total_time_s": float(global_total_time),
        }
    return {"avg_latency_ms": 0.0, "avg_fps": 0.0, "total_frames": int(global_total_frames), "total_time_s": float(global_total_time)}


def _resolve_sequence_frame_map(
    seq_name: str,
    curated_split_dir: Path,
    frame_mappings: dict[str, dict[int, int]] | None,
) -> dict[int, int] | None:
    if frame_mappings is not None and seq_name in frame_mappings:
        return frame_mappings[seq_name]
    if not curated_split_dir.is_dir():
        return None
    matches = sorted(curated_split_dir.rglob(f"{seq_name}.review.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 review JSON for {seq_name!r} in {curated_split_dir}, found {len(matches)}"
        )
    review = json.loads(matches[0].read_text(encoding="utf-8"))
    return build_frame_mapping(review)


def run_yolo_inference_for_tracking(
    model_path: Path,
    seqmaps_dir: Path,
    local_hf_dir: Path,
    yolo_out_dir: Path,
    video_extensions: tuple[str, ...],
    *,
    imgsz: int = 1280,
    conf: float = 0.01,
    device: str | int | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    annotation_version: str = "v1",
    review_dir: Path | None = None,
    frame_mappings: dict[str, dict[int, int]] | None = None,
) -> None:
    """Runs YOLO detection with rect=False and curated frame-mapping alignment for tracking."""
    model = (
        YOLO(str(model_path))
        if Path(model_path).suffix.lower() == ".pt"
        else YOLO(str(model_path), task="detect")
    )
    curated_base = (
        Path(review_dir)
        if review_dir is not None
        else Path(local_hf_dir) / "labels_curated" / annotation_version
    )

    for split in splits:
        seqmap_file = Path(seqmaps_dir) / f"drone-{split}.txt"
        if not seqmap_file.is_file():
            continue

        split_video_dir = Path(local_hf_dir) / split
        output_split_dir = Path(yolo_out_dir) / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        curated_split_dir = curated_base / split
        sequences = [line.strip() for line in seqmap_file.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]

        log.info("Running YOLO inference on '%s' split (%d sequences)", split, len(sequences))
        for seq_name in sequences:
            video_path = None
            for ext in video_extensions:
                temp_path = split_video_dir / f"{seq_name}{ext}"
                if temp_path.exists():
                    video_path = temp_path
                    break

            if not video_path:
                continue

            frame_map = _resolve_sequence_frame_map(seq_name, curated_split_dir, frame_mappings)
            predict_kwargs: dict[str, object] = {
                "stream": True,
                "imgsz": imgsz,
                "conf": conf,
                "rect": False,
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = device

            results = model(str(video_path), **predict_kwargs)
            detections = []

            for source_frame_idx, result in enumerate(results, start=1):
                if frame_map is not None:
                    if source_frame_idx not in frame_map:
                        continue
                    frame_id = frame_map[source_frame_idx]
                else:
                    frame_id = source_frame_idx

                boxes = result.boxes
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        score = confs[i]
                        cls = int(clss[i])
                        detections.append([frame_id, -1, x1, y1, x2 - x1, y2 - y1, score, cls, -1])

            out_file = output_split_dir / f"{seq_name}.txt"
            if len(detections) > 0:
                np.savetxt(out_file, np.array(detections), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d')
            else:
                out_file.write_text("", encoding="utf-8")