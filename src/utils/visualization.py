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


def plot_dataset_geometry_analysis(df_images: pd.DataFrame, df_bboxes: pd.DataFrame, split_name: str) -> None:
    """Plots a 2x2 grid containing detailed original sizes and bounding box geometries."""
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Dataset Geometry & Scale Analysis — Split: '{split_name.upper()}'", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_images["aspect_ratio"], bins=30, color="#4A90D9", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title("Original Image Aspect Ratios (W/H)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Aspect Ratio")
    ax1.set_ylabel("Number of Images")
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_bboxes["pixel_area"], bins=80, color="#E67E22", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title("Target Absolute Areas (pixels²)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("BBox Area (px²)")
    ax2.set_ylabel("Number of Targets")
    ax2.grid(True, alpha=0.2)
    median_area = df_bboxes["pixel_area"].median()
    ax2.axvline(median_area, color="red", linestyle="--", label=f"Median Area = {median_area:.1f} px²")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df_bboxes["relative_area_pct"], bins=80, color="#2ECC71", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax3.set_title("Target Relative Area (% of total image pixels)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Percentage of Frame Area (%)")
    ax3.set_ylabel("Number of Targets")
    ax3.grid(True, alpha=0.2)
    median_pct = df_bboxes["relative_area_pct"].median()
    ax3.axvline(median_pct, color="red", linestyle="--", label=f"Median % = {median_pct:.4f}%")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_bboxes["pixel_width"], df_bboxes["pixel_height"], color="#9B59B6", alpha=0.5, edgecolors="none", s=15)
    ax4.set_title("Target Dimensions Scatter Plot", fontsize=11, fontweight="bold")
    ax4.set_xlabel("BBox Pixel Width (px)")
    ax4.set_ylabel("BBox Pixel Height (px)")
    ax4.grid(True, alpha=0.2)

    max_dim = max(df_bboxes["pixel_width"].max(), df_bboxes["pixel_height"].max())
    ax4.plot([0, max_dim], [0, max_dim], color="black", linestyle="--", alpha=0.5, label="Square Ratio (1:1)")
    ax4.legend()

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


def plot_theta_analysis_dashboard(
    dist_intra: np.ndarray,
    dist_inter: np.ndarray,
    theta_range: np.ndarray,
    kept_fractions: np.ndarray
) -> float:
    """Plots and saves the comprehensive 4-panel threshold analysis dashboard."""
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Analysis of θ Thresholds for k-Center Greedy",
        fontsize=14,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # --- Plot 1: Histogram of intra-video distances ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(dist_intra, bins=100, color="#4A90D9", alpha=0.8, edgecolor="none")
    ax1.set_title("Distances Within the Same Video (θ₁)", fontsize=11, weight="bold")
    ax1.set_xlabel("Cosine distance")
    ax1.set_ylabel("Number of pairs")

    p10 = np.percentile(dist_intra, 10)
    p25 = np.percentile(dist_intra, 25)
    med = np.median(dist_intra)
    ax1.axvline(p10, color="red", linestyle="--", linewidth=1.2, label=f"p10={p10:.3f}")
    ax1.axvline(p25, color="orange", linestyle="--", linewidth=1.2, label=f"p25={p25:.3f}")
    ax1.axvline(med, color="green", linestyle="--", linewidth=1.2, label=f"median={med:.3f}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Histogram of inter-video distances ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(dist_inter, bins=100, color="#E67E22", alpha=0.8, edgecolor="none")
    ax2.set_title("Distances Between Different Videos (θ₂)", fontsize=11, weight="bold")
    ax2.set_xlabel("Cosine distance")
    ax2.set_ylabel("Number of pairs")

    p25_inter = np.percentile(dist_inter, 25)
    med_inter = np.median(dist_inter)
    p75_inter = np.percentile(dist_inter, 75)
    ax2.axvline(p25_inter, color="orange", linestyle="--", linewidth=1.2, label=f"p25={p25_inter:.3f}")
    ax2.axvline(med_inter, color="green", linestyle="--", linewidth=1.2, label=f"median={med_inter:.3f}")
    ax2.axvline(p75_inter, color="purple", linestyle="--", linewidth=1.2, label=f"p75={p75_inter:.3f}")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # --- Plot 3: Elbow Curve ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(theta_range, kept_fractions * 100, color="#2ECC71", linewidth=2)
    ax3.set_title("Elbow Curve: θ₁ vs % of Frames Retained", fontsize=11, weight="bold")
    ax3.set_xlabel("θ₁")
    ax3.set_ylabel("% of frames retained")
    ax3.grid(True, alpha=0.3)

    second_deriv = np.gradient(np.gradient(kept_fractions))
    elbow_idx = np.argmax(np.abs(second_deriv))
    elbow_theta = theta_range[elbow_idx]
    ax3.axvline(elbow_theta, color="red", linestyle="--", linewidth=1.5, label=f"elbow ≈ {elbow_theta:.3f}")
    ax3.legend(fontsize=9)

    # --- Plot 4: Combined distributions comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(dist_intra, bins=80, color="#4A90D9", alpha=0.6, density=True, label="within the video (θ₁)")
    ax4.hist(dist_inter, bins=80, color="#E67E22", alpha=0.6, density=True, label="between videos (θ₂)")
    ax4.set_title("Comparison of Distributions", fontsize=11, weight="bold")
    ax4.set_xlabel("Cosine distance")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    plt.show()

    return elbow_theta



def plot_post_kcenter_theta2_dashboard(dists: np.ndarray) -> None:
    """Plots and saves the histogram and Cumulative Distribution Function (CDF) for theta2."""
    p10 = np.percentile(dists, 10)
    p25 = np.percentile(dists, 25)
    med = np.median(dists)
    p75 = np.percentile(dists, 75)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("Analysis of θ₂ — Inter-Video Distances (Post Stage 2 k-Center)",
                 fontsize=13, fontweight="bold")

    # Subplot 1: Distance Histogram
    ax = axes[0]
    ax.hist(dists, bins=100, color="#E67E22", alpha=0.85, edgecolor="none")
    ax.axvline(p10, color="red", linestyle="--", lw=1.5, label=f"p10={p10:.3f}")
    ax.axvline(p25, color="orange", linestyle="--", lw=1.5, label=f"p25={p25:.3f}")
    ax.axvline(med, color="green", linestyle="--", lw=1.5, label=f"median={med:.3f}")
    ax.axvline(p75, color="purple", linestyle="--", lw=1.5, label=f"p75={p75:.3f}")
    ax.set_title("Inter-Video Distance Distribution")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Number of pairs")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    # Subplot 2: Cumulative Distribution Function (CDF)
    ax2 = axes[1]
    sorted_d = np.sort(dists)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax2.plot(sorted_d, cdf * 100, color="#E67E22", lw=2)
    for thresh, color, label in [
        (p10, "red", f"p10={p10:.3f}"),
        (p25, "orange", f"p25={p25:.3f}"),
        (med, "green", f"median={med:.3f}"),
    ]:
        ax2.axvline(thresh, color=color, linestyle="--", lw=1.5, label=label)
    ax2.set_title("CDF — % of Pairs with Distance ≤ θ₂")
    ax2.set_xlabel("θ₂")
    ax2.set_ylabel("% of pairs")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*50}\nRECOMMENDATIONS FOR θ₂\n{'='*50}")
    print(f"  p10 = {p10:.4f}  — Highly conservative (removes only identical background setups)")
    print(f"  p25 = {p25:.4f}  — Moderate (excellent target starting point)")
    print(f"  median = {med:.4f}  — Aggressive (keeps only extremely distinct scenes)")
    print(f"\nIf the left tail of your histogram starts above 0.35 - 0.40:")
    print("  -> Your videos are already highly distinct.")
    print("  -> Stage 3 global deduplication is likely NOT necessary.")



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
    from src.processing.yolo import manual_eigencam, run_eigencam_gradcam, preprocess_image_tensor
    
    img_path = str(fp["path"])
    img_rgb, _, tensor = preprocess_image_tensor(img_path, device)
    if img_rgb is None:
        return

    cam_map = run_eigencam_gradcam(pytorch_model, target_layer, tensor)
    if cam_map is None:
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