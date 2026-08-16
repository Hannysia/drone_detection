import configparser
import logging
import sys
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

try:
    from boxmot.trackers import ByteTrack, OcSort
except ImportError:
    try:
        from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
        try:
            from boxmot.trackers.bbox.ocsort.ocsort import OcSort
        except ImportError:
            from boxmot.trackers.bbox.ocsort.ocsort import OCSort as OcSort
    except ImportError:
        try:
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            try:
                from boxmot.trackers.ocsort.ocsort import OcSort
            except ImportError:
                from boxmot.trackers.ocsort.ocsort import OCSort as OcSort
        except ImportError:
            try:
                from boxmot import BYTETracker as ByteTrack
                from boxmot import OCSORT as OcSort
            except ImportError as e:
                raise ImportError(
                    "Failed to import trackers. Please check your BoxMOT installation. "
                    f"Original error: {e}"
                )

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
) -> None:
    """Runs tracking and measures high-precision tracker execution time (latency and FPS)."""
    split_dir = dataset_base_dir / f"drone-{split_name}"
    if not split_dir.exists():
        print(f"Error: Split directory {split_dir} does not exist.")
        return
        
    sequences = [d for d in split_dir.iterdir() if d.is_dir()]
    global_total_time = 0.0
    global_total_frames = 0
    
    for seq in sequences:
        seq_name = seq.name
        
        config = configparser.ConfigParser()
        config.read(seq / "seqinfo.ini")
        width = int(config['Sequence']['imWidth'])
        height = int(config['Sequence']['imHeight'])
        fps = int(config['Sequence']['frameRate'])
        seq_length = int(config['Sequence']['seqLength']) 
        
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
            open(res_file, 'w').close()
            
    if global_total_frames > 0:
        global_latency_ms = (global_total_time / global_total_frames) * 1000
        global_fps = global_total_frames / global_total_time
        print(f"===> [{tracker_name.upper()}] {run_dir_name.upper()} | Config: {tracker_config} | Avg Latency: {global_latency_ms:.3f} ms/frame | Avg FPS: {global_fps:.1f}")


def run_yolo_inference_for_tracking(
    model_path: Path, 
    seqmaps_dir: Path, 
    local_hf_dir: Path, 
    yolo_out_dir: Path, 
    video_extensions: tuple[str]
) -> None:
    """Runs YOLO engine GPU inference on val and test splits and saves detections as flat files."""
    model = YOLO(str(model_path), task="detect")
    splits = ["val", "test"]

    for split in splits:
        split_video_dir = local_hf_dir / split
        output_split_dir = yolo_out_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        seqmap_file = seqmaps_dir / f"drone-{split}.txt"
        with open(seqmap_file, "r", encoding="utf-8") as f:
            sequences = f.read().splitlines()[1:]

        print(f"\n--- Running YOLO GPU inference on '{split}' split ---")
        for seq_name in sequences:
            video_path = None
            for ext in video_extensions:
                temp_path = split_video_dir / f"{seq_name}{ext}"
                if temp_path.exists():
                    video_path = temp_path
                    break

            if not video_path:
                continue

            print(f"Inference on: {video_path.name}")
            results = model(str(video_path), stream=True, conf=0.01, verbose=False)
            detections = []

            for frame_idx, result in enumerate(results):
                frame_id = frame_idx + 1
                boxes = result.boxes
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = confs[i]
                        cls = int(clss[i])
                        detections.append([frame_id, -1, x1, y1, x2 - x1, y2 - y1, conf, cls, -1])

            out_file = output_split_dir / f"{seq_name}.txt"
            if len(detections) > 0:
                np.savetxt(out_file, np.array(detections), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d')
            else:
                open(out_file, 'w').close()