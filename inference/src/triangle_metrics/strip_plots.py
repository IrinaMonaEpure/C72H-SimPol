from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_country_strip_plots(
    input_csv=(
        "inference/output/"
        "original_and_one_variable_flipped_metrics.csv"
    ),
    output_folder=(
        "inference/output/strip_plots"
    ),
):
    input_csv = Path(input_csv)
    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Load individual network data
    # ----------------------------------------------------------
    df = pd.read_csv(input_csv)

    # ----------------------------------------------------------
    # Properties to plot
    # ----------------------------------------------------------
    properties = {
        "number_of_edges": "Number of edges",
        "ratio_imbalanced": "Ratio of imbalanced triangles",
        "edge_weight_sum": "Sum of absolute edge weights",
    }

    # Keep country ordering consistent across all plots
    countries = sorted(df["country"].unique())

    country_positions = {
        country: i
        for i, country in enumerate(countries)
    }

    # ----------------------------------------------------------
    # Separate original and flipped networks
    # ----------------------------------------------------------
    flipped_df = df[
        df["network_type"] == "one_variable_flipped"
    ]

    original_df = df[
        df["network_type"] == "original"
    ]

    # ----------------------------------------------------------
    # Create one plot per property
    # ----------------------------------------------------------
    for property_name, ylabel in properties.items():

        # Taller than before to give the points and legend
        # more vertical space.
        fig, ax = plt.subplots(
            figsize=(14, 8),
        )

        # Deterministic jitter
        rng = np.random.default_rng(42)

        # ======================================================
        # One-variable flipped networks
        # ======================================================
        for i, country in enumerate(countries):

            country_data = flipped_df[
                flipped_df["country"] == country
            ]

            x_position = country_positions[country]

            jitter = rng.uniform(
                -0.18,
                0.18,
                size=len(country_data),
            )

            x_values = (
                np.full(
                    len(country_data),
                    x_position,
                    dtype=float,
                )
                + jitter
            )

            ax.scatter(
                x_values,
                country_data[property_name],
                s=25,
                alpha=0.65,
                color="blue",
                label=(
                    "1-variable flipped networks"
                    if i == 0
                    else None
                ),
            )

        # ======================================================
        # Original networks
        # ======================================================
        original_x = [
            country_positions[country]
            for country in original_df["country"]
        ]

        ax.scatter(
            original_x,
            original_df[property_name],
            s=45,
            color="red",
            label="Original network",
            zorder=3,
        )

        # ------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------
        ax.set_xticks(
            range(len(countries))
        )

        ax.set_xticklabels(
            countries,
            rotation=45,
            ha="right",
        )

        ax.set_xlabel(
            "Country"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{ylabel} across original and "
            "one-variable flipped belief networks"
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        # Always keep legend in the same location
        ax.legend(
            loc="upper right",
        )

        # Add some vertical padding so the highest points
        # are less likely to overlap with the legend.
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        ax.set_ylim(
            y_min,
            y_max + 0.10 * y_range,
        )

        fig.tight_layout()

        # ------------------------------------------------------
        # Save
        # ------------------------------------------------------
        output_path = (
            output_folder
            / f"{property_name}_strip_plot.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    plot_country_strip_plots()