import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.config import MANIFESTS_DIR
from src.processing.video_ablation import (
    ABLATION_MODES,
    ABLATION_TRACKERS,
    DEFAULT_VAL_LOST_FRAMES,
    build_openvino_detector_pair,
    calibrate_trackers_and_roi_on_val,
    load_curated_ablation_sequences,
    prepare_curated_mot_ground_truth,
    resolve_default_openvino_models,
    run_video_ablation_on_test,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenVINO INT8 tracker calibration (val) and adaptive ROI ablation (test).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "prepare-gt", "val", "test"),
        default="all",
        help="Execution stage: prepare-gt, val calibration, test ablation, or all.",
    )
    parser.add_argument("--modes", nargs="+", choices=ABLATION_MODES, default=list(ABLATION_MODES))
    parser.add_argument("--trackers", nargs="+", choices=ABLATION_TRACKERS, default=list(ABLATION_TRACKERS))
    parser.add_argument("--lost-frames-grid", nargs="+", type=int, default=list(DEFAULT_VAL_LOST_FRAMES))
    parser.add_argument("--model-1280", type=Path, default=None)
    parser.add_argument("--model-640", type=Path, default=None)
    parser.add_argument("--quantization-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--hf-dir", type=Path, default=None)
    parser.add_argument("--mot-dataset-dir", type=Path, default=None)
    parser.add_argument("--seqmaps-dir", type=Path, default=None)
    parser.add_argument("--ablation-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=MANIFESTS_DIR / "baseline" / "v1" / "main.csv")
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--det-conf", type=float, default=0.10)
    parser.add_argument("--seed-conf", type=float, default=0.25)
    parser.add_argument("--roi-crop-size", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-prepare-gt", action="store_true")
    return parser


def _parse_and_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.roi_crop_size <= 0:
        parser.error("--fps and --roi-crop-size must be positive")
    if not 0.0 < args.det_conf <= 1.0 or not 0.0 < args.seed_conf <= 1.0:
        parser.error("--det-conf and --seed-conf must be in (0, 1]")
    if any(v < 0 for v in args.lost_frames_grid):
        parser.error("--lost-frames-grid must contain non-negative integers")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_and_validate_args(argv)
    dataset_root = args.dataset_root if args.dataset_root is not None else config.FRAME_BASELINE_DIR / "v1"
    hf_dir = args.hf_dir if args.hf_dir is not None else config.LOCAL_HF_DATA_DIR
    mot_dataset_dir = args.mot_dataset_dir if args.mot_dataset_dir is not None else config.MOT_DATASET_DIR
    seqmaps_dir = args.seqmaps_dir if args.seqmaps_dir is not None else mot_dataset_dir / "seqmaps"
    ablation_dir = args.ablation_dir if args.ablation_dir is not None else config.WORKING_DIR / "video_ablation"

    if args.no_source_check:
        source_path = None
    elif args.source_path is not None:
        source_path = args.source_path
    else:
        source_path = MANIFESTS_DIR / "sources" / "hf.json"

    default_1280, default_640 = resolve_default_openvino_models(args.quantization_dir)
    model_1280 = Path(args.model_1280) if args.model_1280 is not None else default_1280
    model_640 = Path(args.model_640) if args.model_640 is not None else default_640

    reports_dir = ablation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "video_ablation.log"
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_logger.addHandler(file_handler)

    try:
        if not args.skip_prepare_gt and args.stage in ("all", "prepare-gt"):
            prepare_curated_mot_ground_truth(
                hf_dir,
                dataset_root,
                mot_dataset_dir,
                seqmaps_dir,
                splits=("val", "test"),
                manifest_path=args.manifest_path,
                source_path=source_path,
                fps=float(args.fps),
            )
        if args.stage == "prepare-gt":
            return 0

        det_1280, det_640 = build_openvino_detector_pair(
            model_1280,
            model_640,
            det_conf=args.det_conf,
        )
        metadata_path = reports_dir / "video_ablation_metadata.json"

        best_configs: dict[str, dict[str, float | int]] | None = None
        best_lost_frames = int(args.lost_frames_grid[0])

        if args.stage in ("all", "val"):
            val_seqs = load_curated_ablation_sequences(
                dataset_root,
                seqmaps_dir,
                split="val",
                manifest_path=args.manifest_path,
                source_path=source_path,
            )
            val_res = calibrate_trackers_and_roi_on_val(
                val_seqs,
                det_1280,
                det_640,
                mot_dataset_dir,
                ablation_dir,
                trackers=tuple(dict.fromkeys(args.trackers)),
                lost_frames_candidates=tuple(dict.fromkeys(args.lost_frames_grid)),
                fps=args.fps,
                det_conf=args.det_conf,
                seed_conf=args.seed_conf,
                roi_crop_size=args.roi_crop_size,
                resume=args.resume,
            )
            best_configs = val_res["best_tracker_configs"]
            best_lost_frames = int(val_res["best_max_lost_frames"])
            meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            meta.update({
                "tracker_configs": best_configs,
                "max_lost_frames": best_lost_frames,
                "model_1280": str(model_1280),
                "model_640": str(model_640),
            })
            metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            log.info("Val calibration complete. Best configs=%s | max_lost_frames=%d", best_configs, best_lost_frames)

        if args.stage in ("all", "test"):
            if best_configs is None:
                if not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"Calibration metadata missing at {metadata_path}. Run --stage val first."
                    )
                saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                best_configs = saved_meta["tracker_configs"]
                best_lost_frames = int(saved_meta["max_lost_frames"])

            test_seqs = load_curated_ablation_sequences(
                dataset_root,
                seqmaps_dir,
                split="test",
                manifest_path=args.manifest_path,
                source_path=source_path,
            )
            test_res = run_video_ablation_on_test(
                test_seqs,
                det_1280,
                det_640,
                mot_dataset_dir,
                ablation_dir,
                tracker_configs=best_configs,
                max_lost_frames=best_lost_frames,
                modes=tuple(dict.fromkeys(args.modes)),
                trackers=tuple(dict.fromkeys(args.trackers)),
                eval_split="test",
                fps=args.fps,
                det_conf=args.det_conf,
                seed_conf=args.seed_conf,
                roi_crop_size=args.roi_crop_size,
                resume=args.resume,
                metadata_extra={
                    "model_1280": str(model_1280),
                    "model_640": str(model_640),
                },
            )
            cols = ["mode", "tracker", "e2e_fps", "e2e_mean_ms", "pct_track_frames", "HOTA", "IDF1", "MOTA", "IDSW", "track_recall_small"]
            df = test_res["results_df"]
            log.info("\n%s", df[[c for c in cols if c in df.columns]].to_string(index=False))
            log.info("Test ablation CSV: %s", test_res["csv_path"])
        return 0
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())