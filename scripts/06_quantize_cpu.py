import argparse
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src import config
from src.config import MANIFESTS_DIR, QUANTIZATION_RESOLUTIONS
from src.processing.quantization import (
    ALL_STAGES,
    DEFAULT_CALIBRATION_IMAGES,
    run_cpu_quantization_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_SUMMARY_COLUMNS = (
    "imgsz",
    "stage",
    "status",
    "size_mb",
    "compute_mean_ms",
    "compute_fps",
    "e2e_mean_ms",
    "e2e_fps",
    "mAP50",
    "AP50",
    "AP50_small",
    "Neg FPR @ R>=0.85 (%)",
    "ap50_delta_vs_fp32",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU quantization (ONNX INT8 & OpenVINO INT8) and threshold-free ablation.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Evaluation and benchmark split ('val' for tuning, 'test' for final frozen run).",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(QUANTIZATION_RESOLUTIONS),
        help="Square input sizes to export and benchmark (e.g. 1280 640).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional local path to source .pt checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Extracted dataset root (defaults to FRAME_BASELINE_DIR / 'v1').",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
        help="Manifest CSV path.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Optional extraction provenance JSON (defaults to manifests/sources/hf.json).",
    )
    parser.add_argument(
        "--no-source-check",
        action="store_true",
        help="Disable source.json verification when using a custom dataset root.",
    )
    parser.add_argument(
        "--quantization-dir",
        type=Path,
        default=None,
        help="Output directory for quantized artifacts and reports.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help="Temporary staging workspace directory.",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=DEFAULT_CALIBRATION_IMAGES,
        help="Number of stratified train-only images used for INT8 PTQ calibration.",
    )
    parser.add_argument(
        "--calibrate-method",
        choices=("MinMax", "Entropy", "Percentile"),
        default="MinMax",
        help="ONNX Runtime static calibration method.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-benchmark", type=int, default=100)
    parser.add_argument("--collection-conf", type=float, default=0.01)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
        help="Subset of pipeline stages to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid exported artifacts and skip completed stages from checkpoint metadata.",
    )
    parser.add_argument(
        "--skip-existing-exports",
        action="store_true",
        help="Reuse valid exported models already present on disk without re-exporting.",
    )
    parser.add_argument(
        "--eval-pytorch",
        action="store_true",
        help="Also run full CPU evaluation for baseline PyTorch FP32 weights.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload final INT8 artifacts and split reports to Hugging Face.",
    )
    parser.add_argument(
        "--hf-tag",
        type=str,
        default=None,
        help="Optional dataset repository tag created after --upload.",
    )
    return parser


def _parse_and_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if any(r <= 0 or r % 32 != 0 for r in args.resolutions):
        parser.error("--resolutions must contain positive multiples of 32")
    if args.calibration_images <= 0:
        parser.error("--calibration-images must be positive")
    if args.n_warmup < 0:
        parser.error("--n-warmup must be non-negative")
    if args.n_benchmark <= 0:
        parser.error("--n-benchmark must be positive")
    if not 0.0 < args.collection_conf <= 1.0:
        parser.error("--collection-conf must be in (0, 1]")
    if args.hf_tag and not args.upload:
        parser.error("--hf-tag requires --upload")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_and_validate_args(argv)

    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else config.FRAME_BASELINE_DIR / "v1"
    )
    if args.no_source_check:
        source_path = None
    elif args.source_path is not None:
        source_path = args.source_path
    else:
        source_path = MANIFESTS_DIR / "sources" / "hf.json"

    target_quant_dir = (
        Path(args.quantization_dir)
        if args.quantization_dir is not None
        else config.QUANTIZATION_DIR
    )
    log_path = target_quant_dir / "reports" / args.split / "quantize_cpu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_logger.addHandler(file_handler)

    try:
        result = run_cpu_quantization_ablation(
            dataset_root,
            weights_path=args.weights,
            quantization_dir=target_quant_dir,
            working_dir=args.working_dir,
            resolutions=tuple(args.resolutions),
            stages=tuple(dict.fromkeys(args.stages)),
            eval_split=args.split,
            eval_pytorch=args.eval_pytorch,
            resume=args.resume,
            skip_existing_exports=args.skip_existing_exports,
            max_calibration_images=args.calibration_images,
            calibrate_method=args.calibrate_method,
            seed=args.seed,
            n_warmup=args.n_warmup,
            n_benchmark=args.n_benchmark,
            collection_conf=args.collection_conf,
            manifest_path=args.manifest_path,
            source_path=source_path,
            upload_to_hf=args.upload,
            hf_tag=args.hf_tag,
        )

        df = result["results_df"]
        cols = [col for col in _SUMMARY_COLUMNS if col in df.columns]
        if cols:
            log.info("\n%s", df[cols].to_string(index=False))
        log.info("CSV report: %s", result["csv_path"])
        log.info("Metadata JSON: %s", result["metadata_path"])
        log.info("Log file: %s", log_path)

        failed = result.get("failed_stages") or []
        if failed:
            failed_names = " ".join(sorted({item["stage"] for item in failed}))
            log.error(
                "Some stages failed. Re-run safely with: .venv\\Scripts\\python.exe scripts\\06_quantize_cpu.py --split %s --resume --stages %s",
                args.split,
                failed_names,
            )
            return 1
        return 0
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())