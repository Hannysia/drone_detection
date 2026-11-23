from __future__ import annotations

from pathlib import Path
import threading
from typing import Callable

import gradio as gr
from matplotlib.figure import Figure
import pandas as pd

from src.processing.video_demo import (
    DEMO_MODES,
    DEMO_TRACKERS,
    build_demo_detector_pair,
    load_demo_defaults,
    process_video_file,
)

DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "configs" / "video_ablation_metadata.json"

MODE_CHOICES: list[tuple[str, str]] = [
    ("ROI Static (1280 Search + 640 Track)", "roi_static"),
    ("ROI Kalman (1280 Search + 640 Track)", "roi_kalman"),
    ("Full 1280 (Baseline)", "full_1280"),
    ("Resize 640 (Fast Baseline)", "resize_640"),
]

TRACKER_CHOICES: list[tuple[str, str]] = [
    ("ByteTrack", "bytetrack"),
    ("OC-SORT", "ocsort"),
]

_DETECTOR_LOCK = threading.Lock()
_DETECTOR_CACHE: dict[str, tuple[object, object]] = {}


def get_demo_detector_pair(
    loader: Callable[[], tuple[object, object]] | None = None,
) -> tuple[object, object]:
    """Lazily compiles and caches the 1280/640 OpenVINO INT8 detector pair across requests."""
    if "default" in _DETECTOR_CACHE:
        return _DETECTOR_CACHE["default"]
    with _DETECTOR_LOCK:
        if "default" not in _DETECTOR_CACHE:
            resolved_loader = build_demo_detector_pair if loader is None else loader
            _DETECTOR_CACHE["default"] = resolved_loader()
    return _DETECTOR_CACHE["default"]


def build_summary_dataframe(summary: dict[str, object]) -> pd.DataFrame:
    """Formats run telemetry dictionary into a readable two-column summary table."""
    rows = [
        ("Pipeline Mode", str(summary.get("mode", "-"))),
        ("Tracker", str(summary.get("tracker_name", "-"))),
        ("Resolution", f"{int(summary.get('width', 0))}x{int(summary.get('height', 0))}"),
        ("Source Video FPS", f"{float(summary.get('video_fps', 0.0)):.2f}"),
        ("Processed Frames", str(int(summary.get("total_frames", 0)))),
        ("Search 1280 Frames", str(int(summary.get("search_frames", 0)))),
        ("Track 640 Frames", str(int(summary.get("track_frames", 0)))),
        ("Track 640 Share (%)", f"{float(summary.get('pct_track_frames', 0.0)):.1f}%"),
        ("Search <-> Track Transitions", str(int(summary.get("search_transitions", 0)))),
        ("Mean E2E FPS", f"{float(summary.get('e2e_fps', 0.0)):.2f}"),
        ("Mean Latency (ms)", f"{float(summary.get('e2e_mean_ms', 0.0)):.2f}"),
        ("P50 Latency (ms)", f"{float(summary.get('e2e_p50_ms', 0.0)):.2f}"),
        ("P95 Latency (ms)", f"{float(summary.get('e2e_p95_ms', 0.0)):.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_telemetry_figure(
    per_frame_df: pd.DataFrame,
    summary: dict[str, object],
) -> Figure:
    """Builds a thread-safe Matplotlib figure with per-frame FPS and latency mode timeline."""
    fig = Figure(figsize=(9, 4.4), dpi=110)
    ax_fps = fig.add_subplot(2, 1, 1)
    ax_lat = fig.add_subplot(2, 1, 2, sharex=ax_fps)

    if per_frame_df.empty:
        ax_fps.set_title("Per-Frame Processing Speed (No Frames)")
        fig.tight_layout()
        return fig

    frames = per_frame_df["frame_idx"].to_numpy()
    inst_fps = per_frame_df["inst_fps"].to_numpy()
    latency_ms = per_frame_df["latency_ms"].to_numpy()
    is_track = (per_frame_df["mode_used"] == "track_640").to_numpy()

    ax_fps.plot(frames, inst_fps, color="#475569", linewidth=1.2, alpha=0.7)
    if is_track.any():
        ax_fps.scatter(frames[is_track], inst_fps[is_track], color="#16a34a", s=16, label="TRACK 640")
    if (~is_track).any():
        ax_fps.scatter(frames[~is_track], inst_fps[~is_track], color="#ea580c", s=22, label="SEARCH 1280")

    mean_fps = float(summary.get("e2e_fps", 0.0))
    if mean_fps > 0.0:
        ax_fps.axhline(mean_fps, color="#2563eb", linestyle="--", linewidth=1.1, label=f"Mean ({mean_fps:.1f} FPS)")
    ax_fps.set_ylabel("Instant FPS")
    ax_fps.set_title("Per-Frame Speed & Active Resolution Mode")
    ax_fps.grid(alpha=0.25)
    ax_fps.legend(loc="upper right", fontsize=8)

    bar_colors = ["#16a34a" if flag else "#ea580c" for flag in is_track]
    ax_lat.bar(frames, latency_ms, color=bar_colors, width=0.85, alpha=0.85)
    ax_lat.set_xlabel("Frame Index")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def _extract_video_path(video_input: object) -> Path:
    if isinstance(video_input, dict):
        raw = video_input.get("path") or video_input.get("name")
    else:
        raw = video_input
    if not raw or not str(raw).strip():
        raise gr.Error("Upload a video file before starting inference.")
    path = Path(str(raw))
    if not path.is_file():
        raise gr.Error(f"Uploaded video file is unavailable: {path}")
    return path


def run_demo_inference(
    video_input: object,
    mode: str,
    tracker_name: str,
    max_lost_frames: int,
    search_conf: float,
    track_conf: float,
    seed_conf: float,
    max_frames: int,
    draw_roi_box: bool,
    draw_trail: bool,
    draw_hud: bool,
    progress: object = None,
) -> tuple[str, pd.DataFrame, Figure]:
    """Executes Adaptive ROI video processing for Gradio UI and returns video, table, and chart."""
    source_path = _extract_video_path(video_input)
    if mode not in DEMO_MODES:
        raise gr.Error(f"Unsupported mode: {mode}")
    if str(tracker_name).lower() not in DEMO_TRACKERS:
        raise gr.Error(f"Unsupported tracker: {tracker_name}")

    try:
        search_det, track_det = get_demo_detector_pair()
    except Exception as exc:
        raise gr.Error(f"Failed to load OpenVINO INT8 models: {exc}") from exc

    defaults = load_demo_defaults(DEFAULT_METADATA_PATH)
    tracker_configs = defaults["tracker_configs"]
    tracker_cfg = tracker_configs.get(str(tracker_name).lower())
    frame_limit = int(max_frames) if int(max_frames) > 0 else None

    def _on_progress(done: int, total: int) -> None:
        if progress is not None and total > 0:
            progress((done, total), desc=f"Processing frame {done}/{total}")

    try:
        result = process_video_file(
            source_path,
            mode=mode,
            tracker_name=str(tracker_name).lower(),
            tracker_config=tracker_cfg,
            max_lost_frames=int(max_lost_frames),
            search_conf=float(search_conf),
            track_conf=float(track_conf),
            seed_conf=float(seed_conf),
            roi_crop_size=int(defaults["roi_crop_size"]),
            draw_roi_box=bool(draw_roi_box),
            draw_trail=bool(draw_trail),
            draw_hud=bool(draw_hud),
            max_frames=frame_limit,
            search_detector=search_det,
            track_detector=track_det,
            progress_callback=_on_progress if progress is not None else None,
        )
    except Exception as exc:
        raise gr.Error(f"Video processing failed: {exc}") from exc

    summary_df = build_summary_dataframe(result["summary"])
    fig = build_telemetry_figure(result["per_frame_df"], result["summary"])
    return str(result["output_video_path"]), summary_df, fig


def build_demo_app() -> gr.Blocks:
    """Constructs the Gradio Blocks application for local execution and HF Spaces."""
    defaults = load_demo_defaults(DEFAULT_METADATA_PATH)
    with gr.Blocks(title="UAV Adaptive ROI Tracking (OpenVINO INT8)") as blocks:
        gr.Markdown(
            "## UAV Detection & Adaptive ROI Tracking (OpenVINO INT8)\n"
            "Compares `SEARCH 1280` -> `TRACK 640` ROI cropping (`ROI Static` default, `ROI Kalman` optional) against Full 1280 and Resize 640 baselines on CPU.\n\n"
            "*Note: `ROI Static` and `ROI Kalman` share the same OpenVINO 1280/640 compute profile (~81.5% `640` crops on test); both remain selectable for side-by-side comparison on real videos.*"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video (.mp4 / .mov / .avi)")
                mode_input = gr.Radio(
                    choices=MODE_CHOICES,
                    value="roi_static",
                    label="Inference Mode",
                )
                tracker_input = gr.Radio(
                    choices=TRACKER_CHOICES,
                    value="bytetrack",
                    label="Tracker",
                )
                with gr.Accordion("Parameters & Overlays", open=False):
                    max_lost_input = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=int(defaults["max_lost_frames"]),
                        step=1,
                        label="Max Lost Frames Before 1280 Re-Search",
                    )
                    search_conf_input = gr.Slider(
                        minimum=0.01,
                        maximum=0.90,
                        value=float(defaults["det_conf"]),
                        step=0.01,
                        label="Search Confidence (1280)",
                    )
                    track_conf_input = gr.Slider(
                        minimum=0.01,
                        maximum=0.90,
                        value=float(defaults["det_conf"]),
                        step=0.01,
                        label="Track Confidence (640 ROI)",
                    )
                    seed_conf_input = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=float(defaults["seed_conf"]),
                        step=0.01,
                        label="Target Lock Confidence",
                    )
                    max_frames_input = gr.Slider(
                        minimum=0,
                        maximum=1800,
                        value=300,
                        step=30,
                        label="Max Frames (0 = full video)",
                    )
                    draw_roi_input = gr.Checkbox(value=True, label="Show 640x640 ROI Crop Box")
                    draw_trail_input = gr.Checkbox(value=True, label="Show Trajectory Trail")
                    draw_hud_input = gr.Checkbox(value=True, label="Show Telemetry HUD")

                run_btn = gr.Button("Run Tracking", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Annotated Output Video")
                plot_output = gr.Plot(label="Per-Frame Speed & Mode Timeline")
                summary_output = gr.Dataframe(label="Telemetry Summary", interactive=False)

        run_btn.click(
            fn=run_demo_inference,
            inputs=[
                video_input,
                mode_input,
                tracker_input,
                max_lost_input,
                search_conf_input,
                track_conf_input,
                seed_conf_input,
                max_frames_input,
                draw_roi_input,
                draw_trail_input,
                draw_hud_input,
            ],
            outputs=[video_output, summary_output, plot_output],
        )
    return blocks


demo = build_demo_app()

if __name__ == "__main__":
    demo.launch()