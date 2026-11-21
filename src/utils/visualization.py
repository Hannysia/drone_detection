import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import cv2


log = logging.getLogger(__name__)

def plot_duration_histogram(df: pd.DataFrame) -> None:
    """Plots an interactive Plotly histogram showing video duration distribution."""
    fig = px.histogram(
        df, 
        x="duration_sec", 
        nbins=50,
        title="Video Duration Distribution in Dataset",
        labels={"duration_sec": "Duration (seconds)"},
        hover_data=["filename"],
        color_discrete_sequence=['#3366CC']
    )
    fig.update_layout(
        yaxis_title="Number of videos",
        bargap=0.05,
        template="plotly_white"
    )
    fig.show()


def plot_dataset_distribution_summary(df: pd.DataFrame) -> None:
    """Plots a 1x3 Seaborn countplot grid showing shape, camera, and background distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Dataset Distribution Summary', fontsize=16)

    sns.countplot(data=df, x='Shape', ax=axes[0], palette='viridis', hue='Shape', legend=False)
    axes[0].set_title('Drone Shape Distribution')
    axes[0].set_xlabel('Shape Class')
    axes[0].set_ylabel('Count')

    sns.countplot(data=df, x='Camera', ax=axes[1], palette='plasma', hue='Camera', legend=False)
    axes[1].set_title('Camera Sensor Distribution')
    axes[1].set_xlabel('Sensor Class')
    axes[1].set_ylabel('Count')

    sns.countplot(data=df, x='Background', ax=axes[2], palette='coolwarm', hue='Background', legend=False)
    axes[2].set_title('Background Environment Distribution')
    axes[2].set_xlabel('Environment Class')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_stacked_splits_distribution(df_splits: pd.DataFrame) -> None:
    """Generates and saves a horizontal stacked bar chart showing class splits distribution."""
    pivot_counts = pd.crosstab(df_splits['Class'], df_splits['Split'])
    available_splits = [col for col in ['TRAIN', 'VAL', 'TEST'] if col in pivot_counts.columns]
    pivot_counts = pivot_counts[available_splits]
    
    pivot_counts['Total'] = pivot_counts.sum(axis=1)
    pivot_counts = pivot_counts.sort_values(by='Total', ascending=True)
    plot_data = pivot_counts.drop(columns=['Total'])

    plt.figure(figsize=(12, 10))
    colors = ['#4f46e5', '#f59e0b', '#10b981']
    
    plot_data.plot(
        kind='barh', 
        stacked=True, 
        color=colors[:len(available_splits)], 
        ax=plt.gca(), 
        width=0.75
    )
    
    for i, (_, row) in enumerate(pivot_counts.iterrows()):
        total = int(row['Total'])
        plt.text(total + 1, i, f" {total}", va='center', ha='left', fontsize=9, fontweight='bold')

    plt.title('Video Distribution Across Dataset Splits', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Unique Video Sequences', fontsize=12, labelpad=10)
    plt.ylabel('Combined Metadata Classes', fontsize=12, labelpad=10)
    plt.legend(title='Dataset Split', labels=['Train (~70%)', 'Val (~15%)', 'Test (~15%)'], loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.show()


def plot_grouped_splits_report(df_splits: pd.DataFrame) -> None:
    """Generates a 1x3 countplot report of splits grouped by morphology, sensor, and environment."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = ['#4f46e5', '#f59e0b', '#10b981']
    hue_order = ['TRAIN', 'VAL', 'TEST']

    # morphology
    sns.countplot(
        data=df_splits, x='Morphology', hue='Split', hue_order=hue_order,
        palette=colors, ax=axes[0], edgecolor='black', linewidth=0.7
    )
    axes[0].set_title('a) Distribution by Target Shape', fontsize=17, fontweight='bold', pad=15)
    axes[0].set_xlabel('Target Shape', fontsize=15, fontweight='bold', labelpad=10)
    axes[0].set_ylabel('Number of Video Episodes', fontsize=13, fontweight='bold', labelpad=10)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    legend = axes[0].get_legend()
    if legend:
        legend.remove()

    # camera
    sns.countplot(
        data=df_splits, x='Camera', hue='Split', hue_order=hue_order,
        palette=colors, ax=axes[1], edgecolor='black', linewidth=0.7
    )
    axes[1].set_title('b) Distribution by Sensor Type', fontsize=17, fontweight='bold', pad=15)
    axes[1].set_xlabel('Sensor Type', fontsize=15, fontweight='bold', labelpad=10)
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    legend = axes[1].get_legend()
    if legend:
        legend.remove()

    # background
    sns.countplot(
        data=df_splits, x='Background', hue='Split', hue_order=hue_order,
        palette=colors, ax=axes[2], edgecolor='black', linewidth=0.7
    )
    axes[2].set_title('c) Distribution by Background Type', fontsize=17, fontweight='bold', pad=15)
    axes[2].set_xlabel('Background Type', fontsize=15, fontweight='bold', labelpad=10)
    axes[2].set_ylabel('')
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    
    axes[2].legend(
        title='Dataset Split', title_fontsize=14,
        labels=['Train (~70%)', 'Val (~15%)', 'Test (~15%)'], loc='upper right', fontsize=14
    )

    plt.tight_layout()
    plt.show()



def plot_spatial_heatmap(x_centers: np.ndarray, y_centers: np.ndarray) -> None:
    """Generates a 2D Hexagonal Binning Heatmap of YOLO bounding box centers."""
    fig = plt.figure(figsize=(10, 8))
    
    hb = plt.hexbin(
        x_centers, y_centers, 
        gridsize=30, 
        cmap="YlOrRd", 
        mincnt=1, 
        edgecolors="none"
    )
    plt.colorbar(hb, label="Bounding Box Centers Count")
    
    plt.axvline(0.5, color="black", linestyle=":", alpha=0.5, label="Frame Center")
    plt.axhline(0.5, color="black", linestyle=":")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()  
    
    plt.xlabel("Normalized X Coordinate (Width)")
    plt.ylabel("Normalized Y Coordinate (Height)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.15)
    plt.show()


def plot_frame_level_dashboard(df: pd.DataFrame, shape_counts: pd.Series, camera_counts: pd.Series, bg_counts: pd.Series, combined_counts: pd.Series) -> None:
    """Generates 4-panel visual dashboard of frame dataset properties."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Statistical Analysis of the Final Frame-Level Dataset", fontsize=16, fontweight="bold")

    # Plot 1: Morphology
    sns.barplot(x=shape_counts.index, y=shape_counts.values, ax=axes[0, 0], palette="Blues_r", hue=shape_counts.index, legend=False)
    axes[0, 0].set_title("Drone Morphology Distribution (Positive Samples)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Drone Shape")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(shape_counts.values):
        axes[0, 0].text(i, v + 20, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # Plot 2: Camera Sensors
    sns.barplot(x=camera_counts.index, y=camera_counts.values, ax=axes[0, 1], palette="Oranges_r", hue=camera_counts.index, legend=False)
    axes[0, 1].set_title("Camera Sensor Type Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Camera Type")
    axes[0, 1].set_ylabel("Count")
    for i, v in enumerate(camera_counts.values):
        axes[0, 1].text(i, v + 20, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # Plot 3: Background Terrains
    sns.barplot(x=bg_counts.index, y=bg_counts.values, ax=axes[1, 0], palette="Greens_r", hue=bg_counts.index, legend=False)
    axes[1, 0].set_title("Background Terrain Type Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Background Type")
    axes[1, 0].set_ylabel("Count")
    for i, v in enumerate(bg_counts.values):
        axes[1, 0].text(i, v + 20, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # Plot 4: Top 10 Combined Classes
    sns.barplot(x=combined_counts.values, y=combined_counts.index, ax=axes[1, 1], palette="Purples_r", hue=combined_counts.index, legend=False)
    axes[1, 1].set_title("Top 10 Combined Classes (Metadata Combinations)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Count")
    axes[1, 1].set_ylabel("Combined Class")
    for i, v in enumerate(combined_counts.values):
        axes[1, 1].text(v + 10, i, f"{v:,}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_local_contrast_analysis(df_contrast: pd.DataFrame) -> None:
    """Generates 2-panel analysis plots for Local Weber Contrast."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Local Target-to-Background Weber Contrast Analysis", fontsize=14, fontweight="bold")

    sns.boxplot(
        x="camera", y="contrast", data=df_contrast, ax=axes[0], 
        palette="Oranges_r", hue="camera", legend=False, showfliers=False
    )
    axes[0].set_title("Local Contrast Distribution by Camera Type", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Camera Sensor Type")
    axes[0].set_ylabel("Local Contrast Coefficient")

    df_clean = df_contrast[~df_contrast["background"].isin(["rgb", "unknown"])]
    sns.barplot(
        x="background", y="contrast", hue="camera", data=df_clean, 
        ax=axes[1], palette="Set2", estimator=np.median, errorbar=None         
    )
    axes[1].set_title("Median Local Contrast across Background Types", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Background Terrain Type")
    axes[1].set_ylabel("Median Local Contrast Coefficient")
    axes[1].legend(title="Camera Sensor")

    plt.tight_layout()
    plt.show()




def overlay_heatmap(
    img_rgb: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.4,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    input_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Overlays generated attention maps onto target images."""
    h, w = img_rgb.shape[:2]
    heatmap_src = cam_map
    if crop_box is not None or input_size is not None:
        if crop_box is None or input_size is None:
            raise ValueError("Provide both crop_box and input_size together")
        in_h, in_w = input_size
        top, bottom, left, right = crop_box
        if not (0 <= top < bottom <= in_h and 0 <= left < right <= in_w):
            raise ValueError(f"Invalid crop_box {crop_box} for input_size {input_size}")
        resized_map = cv2.resize(cam_map, (in_w, in_h))
        heatmap_src = resized_map[top:bottom, left:right]

    heatmap = cv2.resize(heatmap_src, (w, h))
    heatmap = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)


def _render_eigencam_triptych(
    img_rgb: np.ndarray,
    overlay: np.ndarray,
    overlay_with_boxes: np.ndarray,
    suptitle: str,
    titles: tuple[str, str, str],
    save_path: Path | None,
    *,
    show: bool = True,
    title_color: str = "black",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(suptitle, fontsize=11, color=title_color, fontweight="bold")
    for axis, image, title in zip(axes, (img_rgb, overlay, overlay_with_boxes), titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    _finalize_figure(fig, Path(save_path) if save_path is not None else None, show, dpi=120)


def plot_eigencam_results(
    img_rgb: np.ndarray,
    overlay: np.ndarray,
    overlay_with_boxes: np.ndarray,
    save_path: Path,
    filename: str,
    boxes_count: int,
    *,
    show: bool = True,
) -> None:
    """Plots and saves side-by-side original image, EigenCAM attention map, and predictions."""
    _render_eigencam_triptych(
        img_rgb,
        overlay,
        overlay_with_boxes,
        f"EigenCAM Activation Map — {filename}",
        ("Original Image", "EigenCAM Attention", f"Predictions ({boxes_count} found)"),
        save_path,
        show=show,
    )


def plot_detections_summary(
    detection_counts: list[int],
    run_name: str,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Generates a bar chart summarizing detections per frame and optionally saves it."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(detection_counts)), detection_counts, color="#4A90D9")
    ax.set_title(f"Detections Per Frame Summary — {run_name}")
    ax.set_xlabel("Sample Frame Index")
    ax.set_ylabel("Bounding Box Count")
    ax.axhline(y=1, color="red", linestyle="--", label="Expected (1)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _finalize_figure(
        fig,
        Path(save_path) if save_path is not None else None,
        show,
        tight_rect=None,
    )


def plot_false_positive_results(
    img_rgb: np.ndarray,
    overlay: np.ndarray,
    overlay_boxes: np.ndarray,
    filename: str,
    n_dets: int,
    max_conf: float,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Renders original image, EigenCAM overlay, and false positive boxes side-by-side."""
    _render_eigencam_triptych(
        img_rgb,
        overlay,
        overlay_boxes,
        f"FALSE POSITIVE — {filename}\nDetections: {n_dets}  Max conf: {max_conf:.3f}",
        ("Original (Empty Background)", "EigenCAM Attention Map", "False Positive BBox (Red)"),
        save_path,
        show=show,
        title_color="red",
    )


def plot_eigencam_pair_comparison(
    img_rgb: np.ndarray,
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    filename: str,
    left_title: str,
    right_title: str,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Plots original frame side-by-side with EigenCAM overlays from two compared runs."""
    _render_eigencam_triptych(
        img_rgb,
        left_panel,
        right_panel,
        f"EigenCAM Comparison — {filename}",
        ("Original Image", left_title, right_title),
        save_path,
        show=show,
    )


def plot_unified_calibration_curves(
    thresholds: np.ndarray,
    precision: list[float],
    recall: list[float],
    f1: list[float],
    f05: list[float],
    neg_fa_rate: list[float],
    opt_th_f1: float,
    opt_th_f05: float,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Plots unified calibration curves (P, R, F1, F0.5 with indicators) and watermark FA rates."""
    fig = plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precision, "g-", label="Precision (P)", linewidth=2)
    plt.plot(thresholds, recall, "b-", label="Recall (R)", linewidth=2)
    plt.plot(thresholds, f1, "y:", label="F1-Score", linewidth=2)
    plt.plot(thresholds, f05, "r-", label="F0.5-Score (Precision Weighted)", linewidth=3)

    plt.axvline(opt_th_f1, color="purple", linestyle="--", label=f"Max F1 Th ({opt_th_f1:.2f})")
    plt.axvline(opt_th_f05, color="black", linestyle="-.", label=f"Max F0.5 Th ({opt_th_f05:.2f})")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("YOLO Threshold Calibration Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, neg_fa_rate, "m-", linewidth=2, label="False Alarm Rate")
    plt.axvline(opt_th_f1, color="purple", linestyle="--")
    plt.axvline(opt_th_f05, color="black", linestyle="-.")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("False Alarm Rate on Negatives (%)")
    plt.title("Watermark / Background Noise Vulnerability Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()

    _finalize_figure(
        fig,
        Path(save_path) if save_path is not None else None,
        show,
        tight_rect=None,
    )


def plot_theta_candidate_analysis(
    results: pd.DataFrame,
    *,
    title: str,
) -> plt.Figure:
    required = {
        "theta",
        "kept_fraction",
        "coverage_p95",
        "coverage_max",
    }

    if results.empty or not required.issubset(results.columns):
        raise ValueError(
            "Expected nonempty theta evaluation results",
        )

    ordered = results.sort_values("theta")
    frontier = results.sort_values("kept_fraction")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    figure.suptitle(title)

    axes[0].plot(
        ordered["theta"],
        ordered["kept_fraction"] * 100,
        marker=".",
    )
    axes[0].set_xlabel("Theta")
    axes[0].set_ylabel("Frames retained (%)")
    axes[0].set_title("Dataset size")

    for column, label in (
        ("coverage_p95", "95th percentile"),
        ("coverage_max", "Maximum"),
    ):
        axes[1].plot(
            frontier["kept_fraction"] * 100,
            frontier[column],
            marker=".",
            label=label,
        )

    axes[1].set_xlabel("Frames retained (%)")
    axes[1].set_ylabel("Distance to nearest retained frame")
    axes[1].set_title("Representation error")
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.3)

    figure.tight_layout()
    return figure


_YOLO_LOSS_SPECS = (
    ("Box Loss", ("train/box_loss",), ("val/box_loss",)),
    ("Classification Loss", ("train/cls_loss",), ("val/cls_loss",)),
    ("DFL Loss", ("train/dfl_loss",), ("val/dfl_loss",)),
)

_YOLO_METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
}


def _resolve_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def _sum_loss_columns(
    history: pd.DataFrame,
    candidates_group: tuple[tuple[str, ...], ...],
) -> pd.Series | None:
    cols = [_resolve_column(history.columns, cands) for cands in candidates_group]
    present = [col for col in cols if col is not None]
    if not present:
        return None
    return history[present].sum(axis=1)


def _finalize_figure(
    fig: plt.Figure,
    save_path: Path | None,
    show: bool,
    *,
    dpi: int = 200,
    tight_rect: tuple[float, float, float, float] | None = (0, 0, 1, 0.94),
) -> None:
    try:
        if tight_rect is not None:
            fig.tight_layout(rect=tight_rect)
        else:
            fig.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(fig)


def load_yolo_training_history(results_csv: Path) -> pd.DataFrame:
    if not results_csv.is_file():
        raise FileNotFoundError(f"YOLO results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"YOLO results CSV is empty: {results_csv}")

    df.columns = [str(column).strip() for column in df.columns]

    known_columns = {
        *(
            name
            for _, train_cands, val_cands in _YOLO_LOSS_SPECS
            for name in (*train_cands, *val_cands)
        ),
        *(name for cands in _YOLO_METRIC_ALIASES.values() for name in cands),
    }
    present_known = [col for col in df.columns if col in known_columns]
    if not present_known:
        raise ValueError(f"No recognized YOLO training columns in: {results_csv}")

    for col in present_known:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "epoch" in df.columns:
        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        if epochs.isna().any():
            raise ValueError(f"Invalid epoch values in: {results_csv}")
        df["epoch"] = epochs.astype(int)
    else:
        df["epoch"] = np.arange(1, len(df) + 1)

    return df.sort_values("epoch").reset_index(drop=True)


def summarize_yolo_training(history: pd.DataFrame) -> dict[str, float | int | str | None]:
    if history.empty:
        raise ValueError("Training history is empty")

    metric_cols = {
        key: _resolve_column(history.columns, cands)
        for key, cands in _YOLO_METRIC_ALIASES.items()
    }
    train_total = _sum_loss_columns(history, tuple(spec[1] for spec in _YOLO_LOSS_SPECS))
    val_total = _sum_loss_columns(history, tuple(spec[2] for spec in _YOLO_LOSS_SPECS))

    p50_col = metric_cols["map50"]
    p50_95_col = metric_cols["map50_95"]

    if (
        p50_col
        and p50_95_col
        and history[p50_col].notna().any()
        and history[p50_95_col].notna().any()
    ):
        fitness = 0.1 * history[p50_col] + 0.9 * history[p50_95_col]
        best_idx = fitness.idxmax()
    elif p50_95_col and history[p50_95_col].notna().any():
        best_idx = history[p50_95_col].idxmax()
    elif p50_col and history[p50_col].notna().any():
        best_idx = history[p50_col].idxmax()
    elif val_total is not None and val_total.notna().any():
        best_idx = val_total.idxmin()
    else:
        best_idx = history.index[-1]

    best_row = history.loc[best_idx]
    last_row = history.iloc[-1]

    def _metric_value(row: pd.Series, key: str) -> float | None:
        col = metric_cols[key]
        if col is None or pd.isna(row[col]):
            return None
        return float(row[col])

    last_train_loss = float(train_total.iloc[-1]) if train_total is not None else None
    last_val_loss = float(val_total.iloc[-1]) if val_total is not None else None
    min_val_loss = float(val_total.min()) if val_total is not None else None
    min_val_loss_epoch = (
        int(history.loc[val_total.idxmin(), "epoch"])
        if val_total is not None and val_total.notna().any()
        else None
    )
    best_train_loss = float(train_total.loc[best_idx]) if train_total is not None else None
    val_train_gap = (
        last_val_loss - last_train_loss
        if last_val_loss is not None and last_train_loss is not None
        else None
    )
    val_loss_rebound = (
        last_val_loss - min_val_loss
        if last_val_loss is not None and min_val_loss is not None
        else None
    )

    best_epoch = int(best_row["epoch"])
    last_epoch = int(last_row["epoch"])
    best_map50_95 = _metric_value(best_row, "map50_95")
    last_map50_95 = _metric_value(last_row, "map50_95")
    best_map50 = _metric_value(best_row, "map50")
    last_map50 = _metric_value(last_row, "map50")

    fit_status = "Balanced fit"
    if (
        val_loss_rebound is not None
        and min_val_loss is not None
        and min_val_loss > 0
        and val_loss_rebound > 0.1 * min_val_loss
        and (
            best_train_loss is None
            or last_train_loss is None
            or last_train_loss <= best_train_loss
        )
    ) or (
        best_map50_95 is not None
        and last_map50_95 is not None
        and (last_epoch - best_epoch) >= 5
        and (best_map50_95 - last_map50_95) > 0.015
    ):
        fit_status = "Overfitting signal"
    elif (
        last_map50 is not None
        and last_map50 < 0.5
        and val_total is not None
        and len(val_total) >= 3
        and float(val_total.iloc[-1]) < float(val_total.iloc[0])
    ):
        fit_status = "Possible underfitting"

    return {
        "epochs_completed": int(len(history)),
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_precision": _metric_value(best_row, "precision"),
        "best_recall": _metric_value(best_row, "recall"),
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "last_precision": _metric_value(last_row, "precision"),
        "last_recall": _metric_value(last_row, "recall"),
        "last_map50": last_map50,
        "last_map50_95": last_map50_95,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "min_val_loss": min_val_loss,
        "min_val_loss_epoch": min_val_loss_epoch,
        "val_train_loss_gap": val_train_gap,
        "val_loss_rebound": val_loss_rebound,
        "fit_status": fit_status,
    }


def plot_yolo_training_curves(
    results_csv: Path,
    run_name: str,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> dict[str, float | int | str | None]:
    history = load_yolo_training_history(results_csv)
    summary = summarize_yolo_training(history)
    epochs = history["epoch"]
    best_epoch = int(summary["best_epoch"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    best_map50_str = f"{summary['best_map50']:.4f}" if summary["best_map50"] is not None else "n/a"
    best_map_str = f"{summary['best_map50_95']:.4f}" if summary["best_map50_95"] is not None else "n/a"
    fig.suptitle(
        f"YOLO Training Monitoring — {run_name} | "
        f"Best Epoch: {best_epoch}/{summary['last_epoch']} | "
        f"mAP50: {best_map50_str} | mAP50-95: {best_map_str} | "
        f"Status: {summary['fit_status']}",
        fontsize=13,
        fontweight="bold",
    )

    for index, (title, train_cands, val_cands) in enumerate(_YOLO_LOSS_SPECS):
        ax = axes[0, index]
        train_col = _resolve_column(history.columns, train_cands)
        val_col = _resolve_column(history.columns, val_cands)
        if train_col:
            ax.plot(epochs, history[train_col], label="Train", color="#2563eb", linewidth=2)
        if val_col:
            ax.plot(epochs, history[val_col], label="Val", color="#ea580c", linewidth=2)
        if train_col and val_col:
            ax.fill_between(
                epochs,
                history[train_col],
                history[val_col],
                color="#94a3b8",
                alpha=0.12,
            )
        ax.axvline(best_epoch, color="#16a34a", linestyle="--", alpha=0.8, label=f"Best ({best_epoch})")
        ax.set_title(f"{title} (Train vs Val)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")

    metric_panels = (
        (
            axes[1, 0],
            "Validation Precision & Recall",
            "Score",
            (
                ("precision", "Precision (B)", "#059669"),
                ("recall", "Recall (B)", "#0284c7"),
            ),
        ),
        (
            axes[1, 1],
            "Validation mAP Curves",
            "mAP",
            (
                ("map50", "mAP@50 (B)", "#7c3aed"),
                ("map50_95", "mAP@50-95 (B)", "#db2777"),
            ),
        ),
    )

    for ax, title, ylabel, series in metric_panels:
        for key, label, color in series:
            col = _resolve_column(history.columns, _YOLO_METRIC_ALIASES[key])
            if col:
                ax.plot(epochs, history[col], label=label, color=color, linewidth=2)
        ax.axvline(best_epoch, color="#16a34a", linestyle="--", alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right")

    gap_ax = axes[1, 2]
    train_total = _sum_loss_columns(history, tuple(spec[1] for spec in _YOLO_LOSS_SPECS))
    val_total = _sum_loss_columns(history, tuple(spec[2] for spec in _YOLO_LOSS_SPECS))
    if train_total is not None:
        gap_ax.plot(epochs, train_total, label="Train Total Loss", color="#2563eb", linewidth=2)
    if val_total is not None:
        gap_ax.plot(epochs, val_total, label="Val Total Loss", color="#ea580c", linewidth=2)
    gap_ax.axvline(best_epoch, color="#16a34a", linestyle="--", alpha=0.8, label=f"Best ({best_epoch})")
    if summary["min_val_loss_epoch"] is not None:
        gap_ax.axvline(
            int(summary["min_val_loss_epoch"]),
            color="#dc2626",
            linestyle=":",
            alpha=0.8,
            label=f"Min Val Loss ({summary['min_val_loss_epoch']})",
        )
    gap_str = f"{summary['val_train_loss_gap']:+.4f}" if summary["val_train_loss_gap"] is not None else "n/a"
    rebound_str = f"{summary['val_loss_rebound']:+.4f}" if summary["val_loss_rebound"] is not None else "n/a"
    p_str = f"{summary['best_precision']:.4f}" if summary["best_precision"] is not None else "n/a"
    r_str = f"{summary['best_recall']:.4f}" if summary["best_recall"] is not None else "n/a"
    gap_ax.text(
        0.03,
        0.04,
        f"Best P / R: {p_str} / {r_str}\nVal-Train Gap: {gap_str}\nVal Rebound: {rebound_str}",
        transform=gap_ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    gap_ax.set_title("Total Loss & Generalization Gap", fontweight="bold")
    gap_ax.set_xlabel("Epoch")
    gap_ax.set_ylabel("Total Loss")
    gap_ax.grid(True, alpha=0.25)
    if gap_ax.get_legend_handles_labels()[0]:
        gap_ax.legend(loc="upper right")

    _finalize_figure(fig, save_path, show)
    return summary


def plot_yolo_runs_comparison(
    runs: dict[str, Path],
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> pd.DataFrame:
    if not runs:
        raise ValueError("Expected at least one run to compare")

    loaded = {}
    summary_rows = []
    for label, csv_path in runs.items():
        history = load_yolo_training_history(csv_path)
        summary = summarize_yolo_training(history)
        loaded[label] = (history, summary)
        summary_rows.append({
            "Run": label,
            "Epochs": summary["epochs_completed"],
            "Best Epoch": summary["best_epoch"],
            "Precision": summary["best_precision"],
            "Recall": summary["best_recall"],
            "mAP50": summary["best_map50"],
            "mAP50-95": summary["best_map50_95"],
            "Val-Train Gap": summary["val_train_loss_gap"],
            "Fit Status": summary["fit_status"],
        })

    comparison_df = pd.DataFrame(summary_rows)
    palette = ("#2563eb", "#ea580c", "#059669", "#7c3aed")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "YOLO Experiment Comparison (Best & Epoch Dynamics)",
        fontsize=14,
        fontweight="bold",
    )

    curve_panels = (
        (axes[0, 0], "Validation Precision", "Score", _YOLO_METRIC_ALIASES["precision"], (0.0, 1.05)),
        (axes[0, 1], "Validation Recall", "Score", _YOLO_METRIC_ALIASES["recall"], (0.0, 1.05)),
        (axes[0, 2], "Validation mAP@50", "mAP", _YOLO_METRIC_ALIASES["map50"], (0.0, 1.05)),
        (axes[1, 0], "Validation mAP@50-95", "mAP", _YOLO_METRIC_ALIASES["map50_95"], (0.0, 1.05)),
        (axes[1, 1], "Validation Classification Loss", "Loss", ("val/cls_loss",), None),
    )

    for ax, title, ylabel, candidates, ylim in curve_panels:
        for idx, (label, (history, summary)) in enumerate(loaded.items()):
            col = _resolve_column(history.columns, candidates)
            if col:
                ax.plot(
                    history["epoch"],
                    history[col],
                    label=f"{label} (best@{summary['best_epoch']})",
                    color=palette[idx % len(palette)],
                    linewidth=2,
                )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")

    bar_ax = axes[1, 2]
    metric_keys = ("Precision", "Recall", "mAP50", "mAP50-95")
    x_positions = np.arange(len(metric_keys))
    n_runs = len(comparison_df)
    width = 0.72 / max(n_runs, 1)

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = [float(row[k]) if pd.notna(row[k]) else 0.0 for k in metric_keys]
        offset = (idx - (n_runs - 1) / 2) * width
        bars = bar_ax.bar(
            x_positions + offset,
            values,
            width=width,
            label=row["Run"],
            color=palette[idx % len(palette)],
        )
        for bar, val in zip(bars, values):
            bar_ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    bar_ax.set_title("Best-Epoch Metrics Comparison", fontweight="bold")
    bar_ax.set_xticks(x_positions)
    bar_ax.set_xticklabels(metric_keys)
    bar_ax.set_ylim(0, 1.12)
    bar_ax.set_ylabel("Score")
    bar_ax.grid(axis="y", alpha=0.25)
    bar_ax.legend(loc="lower right")

    _finalize_figure(fig, save_path, show)
    return comparison_df


def plot_error_analysis_dashboard(
    error_report: dict[str, object],
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plots 2x2 residual error diagnostics across size, background, position, and error modes."""
    required = {"error_types", "by_size", "by_background", "by_position"}
    if not required.issubset(error_report):
        raise ValueError(f"Missing error report keys: {sorted(required - error_report.keys())}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    conf_th = float(error_report.get("conf_threshold", 0.0))
    fig.suptitle(f"Residual Error Analysis (conf={conf_th:.4f})", fontsize=14, fontweight="bold")

    size_df = error_report["by_size"]
    ax_size = axes[0, 0]
    size_vals = [float(v) if pd.notna(v) else 0.0 for v in size_df["Recall"]]
    bars = ax_size.bar(size_df["Scale"], size_vals, color="#2563eb", width=0.55)
    for bar, row in zip(bars, size_df.itertuples(index=False)):
        val = float(row.Recall) if pd.notna(row.Recall) else 0.0
        ax_size.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}\n({int(row.TP)}/{int(row.GT)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_size.set_title("Recall by Target Scale (COCO Area)", fontweight="bold")
    ax_size.set_ylim(0, 1.18)
    ax_size.set_ylabel("Recall")
    ax_size.grid(axis="y", alpha=0.25)

    bg_df = error_report["by_background"]
    ax_bg = axes[0, 1]
    x_bg = np.arange(len(bg_df))
    bg_recall = [float(v) if pd.notna(v) else 0.0 for v in bg_df["Recall"]]
    bg_fa = [float(v) / 100.0 if pd.notna(v) else 0.0 for v in bg_df["Neg FA Rate (%)"]]
    ax_bg.bar(x_bg - 0.18, bg_recall, width=0.36, color="#059669", label="Positive Recall")
    ax_bg.bar(x_bg + 0.18, bg_fa, width=0.36, color="#dc2626", label="Negative FA Rate (frac)")
    ax_bg.set_xticks(x_bg)
    ax_bg.set_xticklabels(bg_df["Group"], rotation=20, ha="right")
    ax_bg.set_ylim(0, 1.15)
    ax_bg.set_ylabel("Rate")
    ax_bg.set_title("Recall & False Alarm Rate by Background", fontweight="bold")
    ax_bg.legend(loc="best")
    ax_bg.grid(axis="y", alpha=0.25)

    pos_df = error_report["by_position"]
    ax_pos = axes[1, 0]
    pos_vals = [float(v) if pd.notna(v) else 0.0 for v in pos_df["Recall"]]
    pos_bars = ax_pos.bar(pos_df["Region"], pos_vals, color=["#7c3aed", "#f59e0b"], width=0.45)
    for bar, row in zip(pos_bars, pos_df.itertuples(index=False)):
        val = float(row.Recall) if pd.notna(row.Recall) else 0.0
        ax_pos.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}\n({int(row.TP)}/{int(row.GT)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_pos.set_title("Spatial Bias Check: Center vs Off-center Recall", fontweight="bold")
    ax_pos.set_ylim(0, 1.18)
    ax_pos.set_ylabel("Recall")
    ax_pos.grid(axis="y", alpha=0.25)

    err_df = error_report["error_types"]
    err_sub = err_df[err_df["Type"] != "TP"]
    ax_err = axes[1, 1]
    colors = ["#dc2626" if "FP" in t else "#ea580c" for t in err_sub["Type"]]
    ax_err.barh(err_sub["Type"], err_sub["Count"], color=colors)
    for idx, count in enumerate(err_sub["Count"]):
        ax_err.text(int(count) + 0.2, idx, str(int(count)), va="center", fontsize=9, fontweight="bold")
    ax_err.set_title("Error Composition (FP & FN Breakdown)", fontweight="bold")
    ax_err.set_xlabel("Count")
    ax_err.grid(axis="x", alpha=0.25)

    _finalize_figure(fig, Path(save_path) if save_path is not None else None, show)


def _draw_gallery_box(
    image: np.ndarray,
    box: list[float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 5, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def plot_error_gallery(
    cases: list[dict],
    *,
    kind: str,
    title: str | None = None,
    max_items: int = 6,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plots a grid of worst False Positive or False Negative frames with annotations."""
    if kind not in ("fp", "fn"):
        raise ValueError(f"Expected kind 'fp' or 'fn', got {kind!r}")
    if not cases:
        log.info("No %s error cases to plot.", kind.upper())
        return

    items = cases[:max_items]
    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    fig.suptitle(title or f"Worst {kind.upper()} Error Cases", fontsize=13, fontweight="bold")
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(flat_axes, items):
        img = cv2.imread(str(item["path"]))
        if img is None:
            raise ValueError(f"Unreadable gallery image: {item['path']}")
        canvas = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if kind == "fp":
            for gt in item.get("gt_boxes", []):
                _draw_gallery_box(canvas, gt, "GT", (0, 220, 0))
            for fp in item["fp_boxes"]:
                _draw_gallery_box(canvas, fp, f"FP {float(fp[4]):.2f}", (255, 0, 0))
            subtitle = f"{item['filename']}\nFP={item['n_fps']} max_conf={item['max_conf']:.2f}"
        else:
            for gt in item["fn_boxes"]:
                _draw_gallery_box(canvas, gt, "MISSED", (0, 255, 0))
            for sub in item.get("subthresh_boxes", []):
                _draw_gallery_box(canvas, sub, f"sub {float(sub[4]):.2f}", (255, 180, 0))
            subtitle = f"{item['filename']}\nMissed={item['n_fns']}"

        ax.imshow(canvas)
        ax.set_title(subtitle, fontsize=9)
        ax.axis("off")

    for ax in flat_axes[len(items):]:
        ax.axis("off")

    _finalize_figure(fig, Path(save_path) if save_path is not None else None, show)


_QUANTIZATION_STAGE_ORDER = (
    "pytorch_fp32",
    "onnx_fp32",
    "onnx_int8",
    "openvino_fp32",
    "openvino_int8",
)
_QUANTIZATION_STAGE_LABELS = {
    "pytorch_fp32": "PyTorch FP32",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8": "ONNX INT8",
    "openvino_fp32": "OpenVINO FP32",
    "openvino_int8": "OpenVINO INT8",
}


def load_quantization_reports(
    quantization_dir: Path,
    *,
    eval_split: str = "val",
) -> tuple[pd.DataFrame, dict]:
    """Loads ablation CSV and metadata JSON for the requested evaluation split."""
    if eval_split not in {"val", "test", "ptq_ablation_val", "ptq_ablation_test"}:
        raise ValueError(f"eval_split must be 'val', 'test', 'ptq_ablation_val' or 'ptq_ablation_test', got {eval_split!r}")

    report_dir = Path(quantization_dir) / "reports" / eval_split
    csv_path = report_dir / "ablation_results.csv"
    metadata_path = report_dir / "quantization_metadata.json"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Quantization ablation CSV not found: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Quantization metadata JSON not found: {metadata_path}")

    results_df = pd.read_csv(csv_path)
    if results_df.empty:
        raise ValueError(f"Quantization ablation CSV is empty: {csv_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return results_df, metadata


def _ordered_quantization_stages(stages_present) -> list[str]:
    known = [s for s in _QUANTIZATION_STAGE_ORDER if s in set(stages_present)]
    extra = sorted(s for s in set(stages_present) if s not in _QUANTIZATION_STAGE_ORDER)
    return known + extra


def plot_quantization_ablation_dashboard(
    results_df: pd.DataFrame,
    metadata: dict | None = None,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plots a 2x2 CPU quantization ablation dashboard across formats and resolutions."""
    required = {
        "imgsz",
        "stage",
        "compute_mean_ms",
        "e2e_mean_ms",
        "e2e_fps",
        "AP50",
        "AP50_small",
        "mAP50",
        "Neg FPR @ R>=0.85 (%)",
        "Recall_small @ Neg FPR<=5%",
    }
    if results_df.empty or not required.issubset(results_df.columns):
        missing = sorted(required - set(results_df.columns))
        raise ValueError(f"Invalid quantization ablation dataframe; missing: {missing}")

    df = results_df.copy()
    stages = _ordered_quantization_stages(df["stage"])
    resolutions = sorted((int(r) for r in df["imgsz"].unique()), reverse=True)
    res_colors = {1280: "#2563eb", 640: "#059669"}
    fallback_palette = ("#7c3aed", "#ea580c", "#0284c7")

    def _res_color(idx: int, imgsz: int) -> str:
        return res_colors.get(imgsz, fallback_palette[idx % len(fallback_palette)])

    eval_split = str((metadata or {}).get("eval_split", "")).upper()
    title_suffix = f" ({eval_split})" if eval_split else ""

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        f"CPU Quantization & Resolution Ablation{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    x_pos = np.arange(len(stages))
    width = 0.72 / max(len(resolutions), 1)
    stage_labels = [_QUANTIZATION_STAGE_LABELS.get(s, s) for s in stages]

    ax_lat = axes[0, 0]
    for idx, imgsz in enumerate(resolutions):
        sub = df[df["imgsz"] == imgsz].set_index("stage")
        e2e_vals = [float(sub.loc[s, "e2e_mean_ms"]) if s in sub.index else 0.0 for s in stages]
        comp_vals = [float(sub.loc[s, "compute_mean_ms"]) if s in sub.index else 0.0 for s in stages]
        fps_vals = [float(sub.loc[s, "e2e_fps"]) if s in sub.index else 0.0 for s in stages]
        offset = (idx - (len(resolutions) - 1) / 2) * width
        color = _res_color(idx, imgsz)
        bars = ax_lat.bar(
            x_pos + offset,
            e2e_vals,
            width=width,
            color=color,
            alpha=0.85,
            label=f"{imgsz} End-to-End ms",
        )
        ax_lat.scatter(
            x_pos + offset,
            comp_vals,
            color="#0f172a",
            marker="D",
            s=36,
            zorder=5,
            label=f"{imgsz} Compute-Only ms" if idx == 0 else None,
        )
        for bar, val, fps in zip(bars, e2e_vals, fps_vals):
            if val > 0:
                ax_lat.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max(e2e_vals) * 0.02,
                    f"{val:.1f}ms\n({fps:.1f}fps)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax_lat.set_title("CPU Latency: End-to-End vs Compute-Only", fontweight="bold")
    ax_lat.set_xticks(x_pos)
    ax_lat.set_xticklabels(stage_labels, rotation=15, ha="right")
    ax_lat.set_ylabel("Latency (ms, lower is better)")
    ax_lat.grid(axis="y", alpha=0.25)
    ax_lat.legend(loc="upper right")

    eval_stages = [s for s in stages if df[df["stage"] == s]["AP50"].notna().any()]
    x_eval = np.arange(len(eval_stages))
    eval_labels = [_QUANTIZATION_STAGE_LABELS.get(s, s) for s in eval_stages]
    ax_acc = axes[0, 1]
    for idx, imgsz in enumerate(resolutions):
        sub = df[df["imgsz"] == imgsz].set_index("stage")
        ap50_vals = [
            float(sub.loc[s, "AP50"]) if s in sub.index and pd.notna(sub.loc[s, "AP50"]) else 0.0
            for s in eval_stages
        ]
        small_vals = [
            float(sub.loc[s, "AP50_small"])
            if s in sub.index and pd.notna(sub.loc[s, "AP50_small"])
            else 0.0
            for s in eval_stages
        ]
        offset = (idx - (len(resolutions) - 1) / 2) * width
        color = _res_color(idx, imgsz)
        bars_ap = ax_acc.bar(
            x_eval + offset,
            ap50_vals,
            width=width * 0.48,
            color=color,
            label=f"{imgsz} AP50",
        )
        bars_sm = ax_acc.bar(
            x_eval + offset + width * 0.48,
            small_vals,
            width=width * 0.48,
            color=color,
            alpha=0.45,
            hatch="//",
            label=f"{imgsz} AP50_small",
        )
        for bar, val in zip(bars_ap, ap50_vals):
            if val > 0:
                ax_acc.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        for bar, val in zip(bars_sm, small_vals):
            if val > 0:
                ax_acc.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_acc.set_title("Threshold-Free Detection Quality (AP50 vs AP50_small)", fontweight="bold")
    ax_acc.set_xticks(x_eval + width * 0.24)
    ax_acc.set_xticklabels(eval_labels, rotation=15, ha="right")
    ax_acc.set_ylim(0, 1.12)
    ax_acc.set_ylabel("AP50")
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend(loc="lower right", ncol=2, fontsize=8)

    ax_safe = axes[1, 0]
    ax_fpr = ax_safe.twinx()
    for idx, imgsz in enumerate(resolutions):
        sub = df[df["imgsz"] == imgsz].set_index("stage")
        rec_sm = [
            float(sub.loc[s, "Recall_small @ Neg FPR<=5%"])
            if s in sub.index and pd.notna(sub.loc[s, "Recall_small @ Neg FPR<=5%"])
            else 0.0
            for s in eval_stages
        ]
        neg_fpr = [
            float(sub.loc[s, "Neg FPR @ R>=0.85 (%)"])
            if s in sub.index and pd.notna(sub.loc[s, "Neg FPR @ R>=0.85 (%)"])
            else np.nan
            for s in eval_stages
        ]
        offset = (idx - (len(resolutions) - 1) / 2) * width
        color = _res_color(idx, imgsz)
        ax_safe.bar(
            x_eval + offset,
            rec_sm,
            width=width,
            color=color,
            alpha=0.75,
            label=f"{imgsz} Recall_small @ Neg FPR<=5%",
        )
        valid_pts = [(pos, fpr) for pos, fpr in zip(x_eval + offset, neg_fpr) if pd.notna(fpr)]
        if valid_pts:
            ax_fpr.plot(
                [p[0] for p in valid_pts],
                [p[1] for p in valid_pts],
                marker="o",
                linestyle="--",
                color="#dc2626" if idx == 0 else "#f59e0b",
                linewidth=1.8,
                label=f"{imgsz} Neg FPR @ R>=0.85 (%)",
            )
    ax_safe.set_title("Small-Target Recall vs Background False Alarms", fontweight="bold")
    ax_safe.set_xticks(x_eval)
    ax_safe.set_xticklabels(eval_labels, rotation=15, ha="right")
    ax_safe.set_ylim(0, 1.1)
    ax_safe.set_ylabel("Recall_small @ Neg FPR<=5% (higher better)")
    ax_fpr.set_ylabel("Neg FPR @ R>=0.85 % (lower better)", color="#b91c1c")
    ax_safe.grid(axis="y", alpha=0.25)
    h1, l1 = ax_safe.get_legend_handles_labels()
    h2, l2 = ax_fpr.get_legend_handles_labels()
    if h1 or h2:
        ax_safe.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    ax_pareto = axes[1, 1]
    stage_colors = {
        "onnx_fp32": "#64748b",
        "onnx_int8": "#2563eb",
        "openvino_fp32": "#f59e0b",
        "openvino_int8": "#059669",
    }
    res_markers = {1280: "o", 640: "s"}
    for _, row in df[df["AP50"].notna()].iterrows():
        imgsz = int(row["imgsz"])
        stage = str(row["stage"])
        fps = float(row["e2e_fps"])
        ap50 = float(row["AP50"])
        ax_pareto.scatter(
            fps,
            ap50,
            color=stage_colors.get(stage, "#7c3aed"),
            marker=res_markers.get(imgsz, "^"),
            s=110,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        ax_pareto.annotate(
            f"{_QUANTIZATION_STAGE_LABELS.get(stage, stage)}\n({imgsz})",
            (fps, ap50),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=8,
        )
    ax_pareto.set_title("Speed-Accuracy Pareto Frontier (CPU End-to-End)", fontweight="bold")
    ax_pareto.set_xlabel("End-to-End FPS (higher is better)")
    ax_pareto.set_ylabel("Threshold-Free AP50 (higher is better)")
    ax_pareto.grid(True, alpha=0.25)

    _finalize_figure(fig, Path(save_path) if save_path is not None else None, show)


def plot_quantization_curves(
    metadata: dict,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plots threshold-free PR, small-recall vs Neg FA curves, and TP confidence spread."""
    stage_details = metadata.get("stage_details") or {}
    if not stage_details:
        raise ValueError("Missing stage_details curves in quantization metadata")

    def _key_sort(name: str) -> tuple[int, int, str]:
        stem, _, res_str = name.rpartition("_")
        res = int(res_str) if res_str.isdigit() else 0
        order_idx = (
            _QUANTIZATION_STAGE_ORDER.index(stem)
            if stem in _QUANTIZATION_STAGE_ORDER
            else len(_QUANTIZATION_STAGE_ORDER)
        )
        return -res, order_idx, name

    stage_colors = {
        "onnx_fp32": "#64748b",
        "onnx_int8": "#2563eb",
        "openvino_fp32": "#f59e0b",
        "openvino_int8": "#059669",
        "pytorch_fp32": "#7c3aed",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(
        "Threshold-Free Quantization Curves & Tracker Confidence Alignment",
        fontsize=14,
        fontweight="bold",
    )

    for key in sorted(stage_details, key=_key_sort):
        curves = stage_details[key].get("curves") or {}
        if not curves:
            continue
        stage_name, _, res_str = key.rpartition("_")
        label = f"{_QUANTIZATION_STAGE_LABELS.get(stage_name, stage_name)} ({res_str})"
        color = stage_colors.get(stage_name, "#0f172a")
        linestyle = "-" if res_str == "1280" else "--"

        axes[0].plot(
            curves.get("recall", []),
            curves.get("precision", []),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        axes[1].plot(
            curves.get("neg_fa_rate", []),
            curves.get("recall_small", []),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )

    axes[0].set_title("Precision-Recall Curve Across Formats", fontweight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=8)

    axes[1].axvline(5.0, color="#dc2626", linestyle=":", linewidth=1.5, label="5% Neg FA Target")
    axes[1].set_title("Small-Target Recall vs Background False Alarms", fontweight="bold")
    axes[1].set_xlabel("Negative Frame False Alarm Rate (%)")
    axes[1].set_ylabel("Small-Target Recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)

    summary_rows = [
        row
        for row in (metadata.get("summary_rows") or [])
        if row.get("tp_conf_median") is not None and pd.notna(row.get("tp_conf_median"))
    ]
    ax_conf = axes[2]
    ax_conf.axhspan(0.3, 0.7, color="#dbeafe", alpha=0.5, label="ByteTrack track_thresh Grid (0.3-0.7)")
    if summary_rows:
        y_labels = [
            f"{_QUANTIZATION_STAGE_LABELS.get(str(r['stage']), r['stage'])} ({int(r['imgsz'])})"
            for r in summary_rows
        ]
        x_idx = np.arange(len(summary_rows))
        medians = np.asarray([float(r["tp_conf_median"]) for r in summary_rows])
        p10s = np.asarray([float(r["tp_conf_p10"]) for r in summary_rows])
        p90s = np.asarray([float(r["tp_conf_p90"]) for r in summary_rows])
        yerr = np.vstack([np.maximum(0.0, medians - p10s), np.maximum(0.0, p90s - medians)])
        ax_conf.errorbar(
            x_idx,
            medians,
            yerr=yerr,
            fmt="o",
            color="#0f172a",
            ecolor="#475569",
            elinewidth=2,
            capsize=5,
            label="TP Conf Median (p10-p90)",
        )
        ax_conf.set_xticks(x_idx)
        ax_conf.set_xticklabels(y_labels, rotation=25, ha="right", fontsize=8)
    ax_conf.set_title("TP Confidence Stability (p10 / Median / p90)", fontweight="bold")
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("Confidence Score")
    ax_conf.grid(axis="y", alpha=0.25)
    ax_conf.legend(loc="lower right", fontsize=8)

    _finalize_figure(fig, Path(save_path) if save_path is not None else None, show)