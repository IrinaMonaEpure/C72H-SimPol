from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_best_triangle_predictors(
    triangle_csv=(
        "../../output/signed_triangle_counts_expanded_2.csv"),
    polarization_csv=(
        "../../../abm/output/polarization_mdl_aggregated.csv"),
    output_path=(
        "../../output/best_triangle_predictors_scatter.png"),
):
    triangle_csv = Path(triangle_csv)
    polarization_csv = Path(polarization_csv)
    output_path = Path(output_path)

    triangle_df = pd.read_csv(triangle_csv)
    polarization_df = pd.read_csv(polarization_csv)

    triangle_df = triangle_df.rename(
        columns={"graph_name": "country"}
    )

    # Best predictor identified for each beta configuration.
    best_predictors = {
        (0.5, 0.5): "three_negative_onnela_intensity_sum",
        (0.5, 2.0): "ratio_three_negative",
        (2.0, 0.5): "ratio_imbalanced",
        (2.0, 2.0): "ratio_two_positive_one_negative",
    }

    metric_labels = {
        "three_negative_onnela_intensity_sum": (
            "Sum of Onnela intensity\n"
            "(3-negative triangles)"
        ),
        "ratio_three_negative": (
            "Ratio of 3-negative triangles"
        ),
        "ratio_imbalanced": (
            "Ratio of imbalanced triangles"
        ),
        "ratio_two_positive_one_negative": (
            "Ratio of 2-positive, 1-negative triangles"
        ),
    }

    metrics_needed = list(
        set(best_predictors.values())
    )

    merged = polarization_df.merge(
        triangle_df[
            ["country"] + metrics_needed
        ],
        on="country",
        how="inner",
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
    )

    axes = axes.flatten()

    for ax, (
        beta_configuration,
        metric,
    ) in zip(
        axes,
        best_predictors.items(),
    ):
        beta_internal, beta_external = (
            beta_configuration
        )

        data = merged[
            (
                merged["beta_internal"]
                == beta_internal
            )
            & (
                merged["beta_external"]
                == beta_external
            )
        ].copy()

        data = data[
            [
                "country",
                metric,
                "polarization_mean",
            ]
        ].dropna()

        x = data[metric].to_numpy(
            dtype=float
        )

        y = data[
            "polarization_mean"
        ].to_numpy(
            dtype=float
        )

        # ------------------------------------------
        # Scatter
        # ------------------------------------------

        ax.scatter(
            x,
            y,
            s=55,
            zorder=3,
        )

        # ------------------------------------------
        # Country labels
        # ------------------------------------------

        for _, row in data.iterrows():
            ax.annotate(
                row["country"],
                (
                    row[metric],
                    row["polarization_mean"],
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

        # ------------------------------------------
        # Linear regression
        # ------------------------------------------

        model = LinearRegression()

        model.fit(
            x.reshape(-1, 1),
            y,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200,
        )

        y_line = model.predict(
            x_line.reshape(-1, 1)
        )

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        pearson_r, pearson_p = pearsonr(
            x,
            y,
        )

        spearman_rho, _ = spearmanr(
            x,
            y,
        )

        r2 = model.score(
            x.reshape(-1, 1),
            y,
        )

        stats_text = (
            f"$r$ = {pearson_r:.2f}\n"
            f"$\\rho$ = {spearman_rho:.2f}\n"
            f"$R^2$ = {r2:.2f}\n"
            f"$p$ = {pearson_p:.3f}"
        )

        ax.text(
            1.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.85,
            },
        )

        # ------------------------------------------
        # Labels
        # ------------------------------------------

        ax.set_xlabel(
            metric_labels[metric]
        )

        ax.set_ylabel(
            "Mean polarization"
        )

        ax.set_title(
            (
                rf"$\beta_{{internal}}={beta_internal}$, "
                rf"$\beta_{{external}}={beta_external}$"
            ),
            fontsize=11,
        )

        ax.grid(
            alpha=0.2,
        )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved scatter plot to: {output_path}"
    )


if __name__ == "__main__":
    plot_best_triangle_predictors()