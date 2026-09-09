from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error


def analyse_average_polarization(
    polarization_csv="abm/output/polarization_mdl_aggregated.csv",
    triangle_csv="inference/output/signed_triangle_counts_expanded_2.csv",
    output_csv="abm/output/triangle_predictors_average_polarization.csv",
    top_n=5,
):
    polarization_csv = Path(polarization_csv)
    triangle_csv = Path(triangle_csv)
    output_csv = Path(output_csv)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    polarization_df = pd.read_csv(polarization_csv)
    triangle_df = pd.read_csv(triangle_csv)

    # ----------------------------------------------------------
    # Average polarization across all beta configurations
    #
    # One resulting value per country.
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
            ),
            polarization_std_across_beta=(
                "polarization_mean",
                "std",
            ),
            n_beta_configurations=(
                "polarization_mean",
                "count",
            ),
        )
    )

    print("\nAverage polarization across beta configurations:")
    print(average_polarization.to_string(index=False))

    # ----------------------------------------------------------
    # Prepare triangle data
    # ----------------------------------------------------------
    if "graph_name" in triangle_df.columns:
        triangle_df = triangle_df.rename(
            columns={"graph_name": "country"}
        )

    # Everything from total_triangles onwards is considered
    # a potential triangle predictor.
    first_metric_index = triangle_df.columns.get_loc(
        "total_triangles"
    )

    triangle_metrics = list(
        triangle_df.columns[first_metric_index:]
    )

    # ----------------------------------------------------------
    # Merge
    # ----------------------------------------------------------
    data = triangle_df.merge(
        average_polarization,
        on="country",
        how="inner",
    )

    print(
        f"\nNumber of countries used: {len(data)}"
    )

    # ----------------------------------------------------------
    # Analyse each triangle metric
    # ----------------------------------------------------------
    rows = []

    for metric in triangle_metrics:

        subset = data[
            [
                "country",
                metric,
                "polarization_mean_across_beta",
            ]
        ].dropna()

        if len(subset) < 3:
            continue

        X = subset[[metric]].to_numpy()
        y = subset[
            "polarization_mean_across_beta"
        ].to_numpy()

        # Skip constant predictors
        if np.all(X == X[0]):
            continue

        # Pearson
        pearson_r, pearson_p = pearsonr(
            X[:, 0],
            y,
        )

        # Spearman
        spearman_rho, spearman_p = spearmanr(
            X[:, 0],
            y,
        )

        # ------------------------------------------------------
        # In-sample linear regression
        # ------------------------------------------------------
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        in_sample_r2 = r2_score(
            y,
            y_pred,
        )

        # ------------------------------------------------------
        # Leave-one-country-out cross-validation
        # ------------------------------------------------------
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

        rows.append(
            {
                "metric": metric,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "in_sample_r2": in_sample_r2,
                "loo_cv_r2": loo_cv_r2,
                "loo_cv_mae": loo_cv_mae,
            }
        )

    results = pd.DataFrame(rows)

    # Same idea as your beta-specific analysis:
    # rank primarily by out-of-sample R².
    results = results.sort_values(
        "loo_cv_r2",
        ascending=False,
    ).reset_index(drop=True)

    # ----------------------------------------------------------
    # Save complete results
    # ----------------------------------------------------------
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_csv,
        index=False,
        float_format="%.6f",
    )

    # ----------------------------------------------------------
    # Print top N
    # ----------------------------------------------------------
    print(
        "\nTop triangle metrics for predicting "
        "average polarization across beta configurations:"
    )

    print(
        results[
            [
                "metric",
                "pearson_r",
                "spearman_rho",
                "in_sample_r2",
                "loo_cv_r2",
                "loo_cv_mae",
            ]
        ]
        .head(top_n)
        .to_string(index=False)
    )

    print(
        f"\nFull results saved to: {output_csv}"
    )

    return results, average_polarization


if __name__ == "__main__":
    results, average_polarization = (
        analyse_average_polarization()
    )