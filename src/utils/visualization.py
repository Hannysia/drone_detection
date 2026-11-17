import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import cv2
import torch


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



def overlay_heatmap(img_rgb: np.ndarray, cam_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlays generated attention maps onto target images."""
    h, w = img_rgb.shape[:2]
    heatmap = cv2.resize(cam_map, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)


def plot_eigencam_results(
    img_rgb: np.ndarray, 
    overlay: np.ndarray, 
    overlay_with_boxes: np.ndarray, 
    save_path: Path, 
    filename: str,
    boxes_count: int
) -> None:
    """Plots and saves side-by-side original image, EigenCAM attention map, and predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"EigenCAM Activation Map — {filename}", fontsize=11, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("EigenCAM Attention")
    axes[1].axis("off")

    axes[2].imshow(overlay_with_boxes)
    axes[2].set_title(f"Predictions ({boxes_count} found)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_detections_summary(detection_counts: list[int], run_name: str) -> None:
    """Generates a bar chart summarizing detections per frame and saves it to output_path."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(detection_counts)), detection_counts, color="#4A90D9")
    ax.set_title(f"Detections Per Frame Summary — {run_name}")
    ax.set_xlabel("Sample Frame Index")
    ax.set_ylabel("Bounding Box Count")
    ax.axhline(y=1, color="red", linestyle="--", label="Expected (1)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()



def plot_false_positive_inline(
    fp: dict, 
    pytorch_model: torch.nn.Module, 
    target_layer: torch.nn.Module, 
    device: torch.device
) -> None:
    """Generates EigenCAM attention overlay and plots the False Positive side-by-side inside the notebook."""
    from src.processing.yolo import manual_eigencam, preprocess_image_tensor
  
    img_path = str(fp["path"])
    img_rgb, _, tensor = preprocess_image_tensor(img_path, device)
    if img_rgb is None:
        return

    cam_map = manual_eigencam(pytorch_model, target_layer, tensor)

    overlay = overlay_heatmap(img_rgb, cam_map)
    overlay_boxes = overlay.copy()

    for (x1, y1, x2, y2), conf in zip(fp["boxes"], fp["confs"]):
        cv2.rectangle(overlay_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(
            overlay_boxes, f"FP {conf:.2f}",
            (int(x1), max(int(y1) - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"FALSE POSITIVE — {fp['path'].name}\n"
        f"Detections: {fp['n_dets']}  Max conf: {fp['max_conf']:.3f}",
        fontsize=11, color="red", fontweight="bold"
    )

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (Empty Background)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("EigenCAM Attention Map")
    axes[1].axis("off")

    axes[2].imshow(overlay_boxes)
    axes[2].set_title("False Positive BBox (Red)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()



def plot_unified_calibration_curves(
    thresholds: np.ndarray, 
    precision: list[float], 
    recall: list[float], 
    f1: list[float], 
    f05: list[float], 
    neg_fa_rate: list[float],
    opt_th_f1: float,
    opt_th_f05: float
) -> None:
    """Plots unified calibration curves (P, R, F1, F0.5 with indicators) and watermark FA rates."""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precision, 'g-', label='Precision (P)', linewidth=2)
    plt.plot(thresholds, recall, 'b-', label='Recall (R)', linewidth=2)
    plt.plot(thresholds, f1, 'y:', label='F1-Score', linewidth=2)
    plt.plot(thresholds, f05, 'r-', label='F0.5-Score (Precision Weighted)', linewidth=3)
    
    plt.axvline(opt_th_f1, color='purple', linestyle='--', label=f'Max F1 Th ({opt_th_f1:.2f})')
    plt.axvline(opt_th_f05, color='black', linestyle='-.', label=f'Max F0.5 Th ({opt_th_f05:.2f})')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('YOLO Threshold Calibration Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, neg_fa_rate, 'm-', linewidth=2, label='False Alarm Rate')
    plt.axvline(opt_th_f1, color='purple', linestyle='--')
    plt.axvline(opt_th_f05, color='black', linestyle='-.')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('False Alarm Rate on Negatives (%)')
    plt.title('Watermark / Background Noise Vulnerability Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()


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