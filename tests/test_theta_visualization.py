import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.utils.visualization import plot_theta_candidate_analysis


def test_theta_plot_preserves_results_and_uses_coverage_columns():
    results = pd.DataFrame({
        "theta": [0.2, 0.0, 0.1],
        "kept_fraction": [0.2, 1.0, 0.5],
        "coverage_p95": [0.15, 0.0, 0.08],
        "coverage_max": [0.19, 0.0, 0.09],
    })
    original = results.copy(deep=True)

    figure = plot_theta_candidate_analysis(
        results,
        title="train/positive",
    )

    try:
        assert len(figure.axes) == 2
        assert list(
            figure.axes[0].lines[0].get_xdata()
        ) == [0.0, 0.1, 0.2]
        assert list(
            figure.axes[0].lines[0].get_ydata()
        ) == [100.0, 50.0, 20.0]
        assert list(
            figure.axes[1].lines[0].get_ydata()
        ) == [0.15, 0.08, 0.0]
        assert list(
            figure.axes[1].lines[1].get_ydata()
        ) == [0.19, 0.09, 0.0]

        pd.testing.assert_frame_equal(results, original)
    finally:
        plt.close(figure)


def test_theta_plot_rejects_empty_results():
    with pytest.raises(ValueError):
        plot_theta_candidate_analysis(
            pd.DataFrame(),
            title="empty",
        )