from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut


def analyse_triangle_predictors(
    triangle_csv=(
        "../../output/signed_triangle_counts_expanded_2.csv"),
    polarization_csv=("../../../abm/output/polarization_mdl_aggregated.csv"),
    output_csv=(
        "../../output/triangle_predictors_of_polarization.csv"
    ),
):
    triangle_csv = Path(triangle_csv)
    polarization_csv = Path(polarization_csv)
    output_csv = Path(output_csv)

    triangle_df = pd.read_csv(triangle_csv)
    polarization_df = pd.read_csv(polarization_csv)

    # -----------------------------------------------
    # Country identifier
    # -----------------------------------------------

    triangle_df = triangle_df.rename(
        columns={
            "graph_name": "country",
        }
    )

    # -----------------------------------------------
    # Use every triangle metric from
    # total_triangles onwards.
    # -----------------------------------------------

    start_column = triangle_df.columns.get_loc(
        "total_triangles"
    )

    triangle_metrics = list(
        triangle_df.columns[start_column:]
    )

    # -----------------------------------------------
    # Merge triangle properties with polarization
    # -----------------------------------------------

    merged = polarization_df.merge(
        triangle_df[
            ["country"] + triangle_metrics
        ],
        on="country",
        how="inner",
    )

    print(
        f"Matched {merged['country'].nunique()} "
        "countries."
    )

    print(
        "Beta configurations:",
        merged[
            [
                "beta_internal",
                "beta_external",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    rows = []

    # -----------------------------------------------
    # Treat each beta configuration independently
    # -----------------------------------------------

    beta_groups = merged.groupby(
        [
            "beta_internal",
            "beta_external",
        ]
    )

    for (
        beta_internal,
        beta_external,
    ), beta_df in beta_groups:

        print(
            f"Analysing beta_internal={beta_internal}, "
            f"beta_external={beta_external}"
        )

        for metric in triangle_metrics:

            data = beta_df[
                [
                    "country",
                    metric,
                    "polarization_mean",
                ]
            ].dropna()

            n = len(data)

            if n < 3:
                continue

            X = (
                data[[metric]]
                .to_numpy(dtype=float)
            )

            y = (
                data["polarization_mean"]
                .to_numpy(dtype=float)
            )

            x = X[:, 0]

            # ---------------------------------------
            # Skip constant predictors
            # ---------------------------------------

            if np.allclose(
                x,
                x[0],
            ):
                continue

            # ---------------------------------------
            # Pearson correlation
            # ---------------------------------------

            pearson_r, pearson_p = pearsonr(
                x,
                y,
            )

            # ---------------------------------------
            # Spearman correlation
            # ---------------------------------------

            spearman_rho, spearman_p = spearmanr(
                x,
                y,
            )

            # ---------------------------------------
            # Ordinary linear regression
            # ---------------------------------------

            model = LinearRegression()

            model.fit(
                X,
                y,
            )

            predictions = model.predict(X)

            in_sample_r2 = r2_score(
                y,
                predictions,
            )

            # ---------------------------------------
            # Leave-one-country-out CV
            # ---------------------------------------

            loo = LeaveOneOut()

            cv_predictions = np.empty(
                len(y),
                dtype=float,
            )

            for train_index, test_index in loo.split(X):

                cv_model = LinearRegression()

                cv_model.fit(
                    X[train_index],
                    y[train_index],
                )

                cv_predictions[test_index] = (
                    cv_model.predict(
                        X[test_index]
                    )
                )

            cv_r2 = r2_score(
                y,
                cv_predictions,
            )

            cv_mae = mean_absolute_error(
                y,
                cv_predictions,
            )

            # ---------------------------------------
            # Save result
            # ---------------------------------------

            rows.append(
                {
                    "beta_internal": (
                        beta_internal
                    ),
                    "beta_external": (
                        beta_external
                    ),
                    "metric": metric,
                    "n_countries": n,

                    "pearson_r": (
                        pearson_r
                    ),
                    "pearson_p": (
                        pearson_p
                    ),

                    "spearman_rho": (
                        spearman_rho
                    ),
                    "spearman_p": (
                        spearman_p
                    ),

                    "regression_coefficient": (
                        model.coef_[0]
                    ),

                    "regression_intercept": (
                        model.intercept_
                    ),

                    "in_sample_r2": (
                        in_sample_r2
                    ),

                    "loo_cv_r2": (
                        cv_r2
                    ),

                    "loo_cv_mae": (
                        cv_mae
                    ),
                }
            )

    results = pd.DataFrame(rows)

    # -----------------------------------------------
    # Rank predictors within each beta configuration
    # by predictive CV R²
    # -----------------------------------------------

    results = results.sort_values(
        [
            "beta_internal",
            "beta_external",
            "loo_cv_r2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"\nSaved results to {output_csv}"
    )

    # -----------------------------------------------
    # Print best predictors
    # -----------------------------------------------

    print("\nBest predictors by beta configuration:\n")

    for (
        beta_internal,
        beta_external,
    ), group in results.groupby(
        [
            "beta_internal",
            "beta_external",
        ]
    ):

        print(
            f"\nbeta_internal={beta_internal}, "
            f"beta_external={beta_external}"
        )

        print(
            group[
                [
                    "metric",
                    "pearson_r",
                    "spearman_rho",
                    "in_sample_r2",
                    "loo_cv_r2",
                    "loo_cv_mae",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

    return results


if __name__ == "__main__":
    results = analyse_triangle_predictors()