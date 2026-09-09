from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict


def plot_average_polarization_correlation(
    polarization_csv="abm/output/polarization_mdl_aggregated.csv",
    triangle_csv="inference/output/signed_triangle_counts_expanded_2.csv",
    output_path="inference/output/average_polarization_ratio_imbalanced_scatter.png",
    metric="ratio_imbalanced",
):
    polarization_csv = Path(polarization_csv)
    triangle_csv = Path(triangle_csv)
    output_path = Path(output_path)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    polarization_df = pd.read_csv(polarization_csv)
    triangle_df = pd.read_csv(triangle_csv)

    if "graph_name" in triangle_df.columns:
        triangle_df = triangle_df.rename(
            columns={"graph_name": "country"}
        )

    # ----------------------------------------------------------
    # Average polarization across beta configurations
    # ----------------------------------------------------------
    average_polarization = (
        polarization_df.groupby(
            "country",
            as_index=False,
        )
        .agg(
            polarization_mean_across_beta=(
                "polarization_mean",
                "mean",
            )
        )
    )

    # ----------------------------------------------------------
    # Merge with triangle metrics
    # ----------------------------------------------------------
    data = triangle_df.merge(
        average_polarization,
        on="country",
        how="inner",
    )

    data = data[
        [
            "country",
            metric,
            "polarization_mean_across_beta",
        ]
    ].dropna()

    x = data[metric].to_numpy()
    y = data["polarization_mean_across_beta"].to_numpy()

    X = x.reshape(-1, 1)

    # ----------------------------------------------------------
    # Correlations
    # ----------------------------------------------------------
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    # ----------------------------------------------------------
    # Linear regression
    # ----------------------------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    in_sample_r2 = r2_score(y, y_pred)

    # ----------------------------------------------------------
    # Leave-one-country-out cross-validation
    # ----------------------------------------------------------
    loo = LeaveOneOut()

    y_pred_cv = cross_val_predict(
        LinearRegression(),
        X,
        y,
        cv=loo,
    )

    loo_cv_r2 = r2_score(
        y,
        y_pred_cv,
    )

    loo_cv_mae = mean_absolute_error(
        y,
        y_pred_cv,
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    ax.scatter(
        x,
        y,
        s=45,
    )

    # Country labels
    for _, row in data.iterrows():
        ax.annotate(
            row["country"],
            (
                row[metric],
                row["polarization_mean_across_beta"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    # ----------------------------------------------------------
    # Regression line
    # ----------------------------------------------------------
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
    )

    # ----------------------------------------------------------
    # Labels
    # ----------------------------------------------------------
    ax.set_xlabel(
        "Ratio of imbalanced triangles"
        if metric == "ratio_imbalanced"
        else metric
    )

    ax.set_ylabel(
        "Mean polarization across beta configurations"
    )

    ax.set_title(
        "Initial network imbalance vs. mean polarization"
    )

    ax.grid(
        alpha=0.2,
    )

    # ----------------------------------------------------------
    # Statistics
    #
    # Put them OUTSIDE the plot so they don't cover countries.
    # ----------------------------------------------------------
    stats_text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman ρ = {spearman_rho:.3f}\n"
        f"In-sample R² = {in_sample_r2:.3f}\n"
        f"LOOCV R² = {loo_cv_r2:.3f}\n"
        f"LOOCV MAE = {loo_cv_mae:.3f}\n"
        f"p = {pearson_p:.3g}"
    )

    ax.text(
        1.03,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )

    # Leave room for statistics on the right
    fig.subplots_adjust(
        right=0.74,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
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

    print(f"Saved plot to: {output_path}")

    print("\nStatistics:")
    print(f"Pearson r:    {pearson_r:.3f}")
    print(f"Pearson p:    {pearson_p:.6g}")
    print(f"Spearman rho: {spearman_rho:.3f}")
    print(f"In-sample R²: {in_sample_r2:.3f}")
    print(f"LOOCV R²:     {loo_cv_r2:.3f}")
    print(f"LOOCV MAE:    {loo_cv_mae:.3f}")


if __name__ == "__main__":
    plot_average_polarization_correlation()