import logging
from pathlib import Path
import pandas as pd
import trackeval

log = logging.getLogger(__name__)

def run_trackeval(
    split_name: str, 
    trackers_to_eval: list[str], 
    dataset_base_dir: Path, 
    trackers_out_dir: Path
) -> None:
    """Runs TrackEval silently, saving all results to disk without console printing."""
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        'PLOT_CURVES': False,
        'PRINT_RESULTS': False,       
        'PRINT_ONLY_COMBINED': True,
        'TIME_PROGRESS': False,       
        'PRINT_CONFIG': False         
    })
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': str(dataset_base_dir),
        'TRACKERS_FOLDER': str(trackers_out_dir),
        'SUB_SPLIT': '',
        'SPLIT_TO_EVAL': split_name,
        'BENCHMARK': 'drone',
        'TRACKERS_TO_EVAL': trackers_to_eval,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'SKIP_SPLIT_FOL': False,
        'PRINT_CONFIG': False
    })
    
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity()
    ]
    
    print(f"Evaluating {len(trackers_to_eval)} configurations on split '{split_name}'...")
    evaluator.evaluate([dataset], metrics)
    print("Evaluation complete. Results saved to disk successfully.")

def parse_trackeval_summary(summary_path: Path) -> dict | None:
    """Parses TrackEval summary file and returns a dictionary of metrics."""
    if not summary_path.exists():
        return None
        
    with open(summary_path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) < 2:
        return None
        
    headers = lines[0].split()
    values = lines[1].split()
    normalized_headers = [h.replace('(%)', '') for h in headers]
    
    metrics = {}
    for h, v in zip(normalized_headers, values):
        try:
            metrics[h] = float(v)
        except ValueError:
            metrics[h] = v
    return metrics



def find_and_display_best_configs(
    trackers_out_dir: Path,
    bytetrack_combinations: list[dict],
    ocsort_combinations: list[dict],
    split_name: str = "val",
    suffix: str = ""
) -> tuple[dict, dict]:
    """Parses TrackEval summaries for all combinations and displays the best configured tracker metrics."""
    best_bt_hota, best_bt_metrics, best_bt_config = -1.0, None, None
    for idx, config in enumerate(bytetrack_combinations):
        summary_file = trackers_out_dir / f"drone-{split_name}" / f"bytetrack{suffix}_run_{idx}" / "pedestrian_summary.txt"
        metrics = parse_trackeval_summary(summary_file)
        if metrics and metrics.get("HOTA", 0.0) > best_bt_hota:
            best_bt_hota = metrics["HOTA"]
            best_bt_metrics, best_bt_config = metrics, config
                
    best_oc_hota, best_oc_metrics, best_oc_config = -1.0, None, None
    for idx, config in enumerate(ocsort_combinations):
        summary_file = trackers_out_dir / f"drone-{split_name}" / f"ocsort{suffix}_run_{idx}" / "pedestrian_summary.txt"
        metrics = parse_trackeval_summary(summary_file)
        if metrics and metrics.get("HOTA", 0.0) > best_oc_hota:
            best_oc_hota = metrics["HOTA"]
            best_oc_metrics, best_oc_config = metrics, config

    if best_bt_metrics or best_oc_metrics:
        print("\n" + "="*115)
        print(f"                 BEST CALIBRATED CONFIGURATIONS ON {split_name.upper()} SPLIT")
        print("="*115)
        comparison_data = []
        for name, m, cfg in [("ByteTrack", best_bt_metrics, best_bt_config), ("OC-SORT", best_oc_metrics, best_oc_config)]:
            if m:
                comparison_data.append({
                    "Tracker": name, 
                    "HOTA": m.get("HOTA", 0.0), 
                    "IDF1": m.get("IDF1", 0.0), 
                    "MOTA": m.get("MOTA", 0.0),
                    "FP": int(m.get("CLR_FP", 0)), 
                    "FN": int(m.get("CLR_FN", 0)), 
                    "IDSW": int(m.get("IDSW", 0))
                })
        df = pd.DataFrame(comparison_data)
        float_format = {"HOTA": "{:.2f}%", "IDF1": "{:.2f}%", "MOTA": "{:.2f}%"}
        
        try:
            from IPython.display import display
            display(df.style.format(float_format).hide(axis='index'))
        except ImportError:
            print(df.to_string(index=False))
            
        print("="*115 + "\n")
        
    return best_bt_config, best_oc_config