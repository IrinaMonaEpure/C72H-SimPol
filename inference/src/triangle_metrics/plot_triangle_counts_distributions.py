from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


METRICS = {
    "ratio_three_negative": (
        "Ratio of 3-negative triangles"
    ),
    "ratio_two_positive_one_negative": (
        "Ratio of 2-positive, 1-negative triangles"
    ),
    "ratio_imbalanced": (
        "Ratio of imbalanced triangles"
    ),
    "three_negative_absolute_weight_sum": (
        "Sum of absolute weights — 3-negative"
    ),
    "two_positive_one_negative_absolute_weight_sum": (
        "Sum of absolute weights — 2-positive, 1-negative"
    ),
    "imbalanced_absolute_weight_sum": (
        "Sum of absolute weights — imbalanced"
    ),
    "three_negative_onnela_intensity_sum": (
        "Sum of Onnela intensity — 3-negative"
    ),
    "two_positive_one_negative_onnela_intensity_sum": (
        "Sum of Onnela intensity — 2-positive, 1-negative"
    ),
    "imbalanced_onnela_intensity_sum": (
        "Sum of Onnela intensity — imbalanced"
    ),
}


def get_edge_weight(
    G,
    u,
    v,
    weight_attribute="weight",
):
    edge_data = G.get_edge_data(u, v)

    if edge_data is None:
        raise KeyError(
            f"No edge found between {u!r} and {v!r}."
        )

    if G.is_multigraph():
        if len(edge_data) != 1:
            raise ValueError(
                f"Multiple edges found between "
                f"{u!r} and {v!r}."
            )

        edge_data = next(
            iter(edge_data.values())
        )

    if weight_attribute not in edge_data:
        raise KeyError(
            f"Edge ({u!r}, {v!r}) has no "
            f"{weight_attribute!r} attribute."
        )

    return float(
        edge_data[weight_attribute]
    )


def calculate_triangle_weight_metrics(
    weights,
):
    """
    Calculate weighted metrics for one triangle.

    Returns
    -------
    absolute_weight_sum
        |w1| + |w2| + |w3|

    onnela_intensity
        (|w1 * w2 * w3|)^(1/3)
    """
    absolute_weights = np.abs(
        np.asarray(
            weights,
            dtype=float,
        )
    )

    absolute_weight_sum = float(
        np.sum(absolute_weights)
    )

    onnela_intensity = float(
        np.prod(absolute_weights)
        ** (1.0 / 3.0)
    )

    return (
        absolute_weight_sum,
        onnela_intensity,
    )


def calculate_graph_triangle_metrics(
    G,
    weight_attribute="weight",
):
    """
    Calculate imbalanced signed-triangle metrics.

    Imbalanced triangles are:
    - 3-negative;
    - 2-positive, 1-negative.
    """
    if G.is_directed():
        raise ValueError(
            "Expected an undirected graph."
        )

    total_triangles = 0

    three_negative_edges = 0
    two_positive_one_negative = 0

    three_negative_absolute_weight_sum = 0.0
    two_positive_one_negative_absolute_weight_sum = 0.0

    three_negative_onnela_intensity_sum = 0.0
    two_positive_one_negative_onnela_intensity_sum = 0.0

    # Ensure each triangle is visited once.
    node_order = {
        node: i
        for i, node in enumerate(
            sorted(
                G.nodes(),
                key=lambda node: (
                    G.degree(node),
                    str(node),
                ),
            )
        )
    }

    forward_neighbors = {
        u: {
            v
            for v in G.neighbors(u)
            if node_order[v] > node_order[u]
        }
        for u in G.nodes()
    }

    for u in G.nodes():
        for v in forward_neighbors[u]:

            common_neighbors = (
                forward_neighbors[u]
                & forward_neighbors[v]
            )

            for w in common_neighbors:

                weights = [
                    get_edge_weight(
                        G,
                        u,
                        v,
                        weight_attribute,
                    ),
                    get_edge_weight(
                        G,
                        u,
                        w,
                        weight_attribute,
                    ),
                    get_edge_weight(
                        G,
                        v,
                        w,
                        weight_attribute,
                    ),
                ]

                total_triangles += 1

                negative_count = sum(
                    weight < 0
                    for weight in weights
                )

                positive_count = sum(
                    weight > 0
                    for weight in weights
                )

                (
                    absolute_weight_sum,
                    onnela_intensity,
                ) = (
                    calculate_triangle_weight_metrics(
                        weights
                    )
                )

                if negative_count == 3:

                    three_negative_edges += 1

                    three_negative_absolute_weight_sum += (
                        absolute_weight_sum
                    )

                    three_negative_onnela_intensity_sum += (
                        onnela_intensity
                    )

                elif (
                    positive_count == 2
                    and negative_count == 1
                ):

                    two_positive_one_negative += 1

                    two_positive_one_negative_absolute_weight_sum += (
                        absolute_weight_sum
                    )

                    two_positive_one_negative_onnela_intensity_sum += (
                        onnela_intensity
                    )

    imbalanced_triangles = (
        three_negative_edges
        + two_positive_one_negative
    )

    imbalanced_absolute_weight_sum = (
        three_negative_absolute_weight_sum
        + two_positive_one_negative_absolute_weight_sum
    )

    imbalanced_onnela_intensity_sum = (
        three_negative_onnela_intensity_sum
        + two_positive_one_negative_onnela_intensity_sum
    )

    if total_triangles > 0:

        ratio_three_negative = (
            three_negative_edges
            / total_triangles
        )

        ratio_two_positive_one_negative = (
            two_positive_one_negative
            / total_triangles
        )

        ratio_imbalanced = (
            imbalanced_triangles
            / total_triangles
        )

    else:
        ratio_three_negative = 0.0
        ratio_two_positive_one_negative = 0.0
        ratio_imbalanced = 0.0

    return {
        "total_triangles": (
            total_triangles
        ),

        "three_negative_edges": (
            three_negative_edges
        ),

        "ratio_three_negative": (
            ratio_three_negative
        ),

        "two_positive_one_negative": (
            two_positive_one_negative
        ),

        "ratio_two_positive_one_negative": (
            ratio_two_positive_one_negative
        ),

        "imbalanced_triangles": (
            imbalanced_triangles
        ),

        "ratio_imbalanced": (
            ratio_imbalanced
        ),

        "three_negative_absolute_weight_sum": (
            three_negative_absolute_weight_sum
        ),

        "two_positive_one_negative_absolute_weight_sum": (
            two_positive_one_negative_absolute_weight_sum
        ),

        "imbalanced_absolute_weight_sum": (
            imbalanced_absolute_weight_sum
        ),

        "three_negative_onnela_intensity_sum": (
            three_negative_onnela_intensity_sum
        ),

        "two_positive_one_negative_onnela_intensity_sum": (
            two_positive_one_negative_onnela_intensity_sum
        ),

        "imbalanced_onnela_intensity_sum": (
            imbalanced_onnela_intensity_sum
        ),
    }


def analyse_flipped_networks(
    network_paths,
    experiment,
    weight_attribute="weight",
):
    """
    Calculate triangle metrics for flipped networks.

    experiment must be either:
        "one_dimension"
        "five_dimension"
    """
    rows = []

    for path in tqdm(
        network_paths,
        desc=f"Analysing {experiment}",
        unit="graph",
    ):
        path = Path(path)

        G = nx.read_graphml(path)

        metrics = (
            calculate_graph_triangle_metrics(
                G,
                weight_attribute=weight_attribute,
            )
        )

        if experiment == "one_dimension":

            # Example:
            # GB_anti_immigration_flipped.graphml

            country = (
                path.stem.split("_", 1)[0]
            )

            suffix = "_flipped"

            name_without_country = (
                path.stem[
                    len(country) + 1:
                ]
            )

            if name_without_country.endswith(
                suffix
            ):
                dimension = (
                    name_without_country[
                        :-len(suffix)
                    ]
                )
            else:
                dimension = (
                    name_without_country
                )

            sample = dimension

        elif experiment == "five_dimension":

            # Example:
            # random_flip/networks/GB/GB_0.graphml

            country = path.parent.name
            sample = path.stem

        else:
            raise ValueError(
                "experiment must be "
                "'one_dimension' or "
                "'five_dimension'"
            )

        rows.append(
            {
                "country": country,
                "sample": sample,
                "file_name": path.name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def load_baseline_metrics(
    baseline_csv,
):
    """
    Load triangle metrics for the original networks.

    Expects the new signed_triangle_counts_expanded.csv.
    """
    baseline_df = pd.read_csv(
        baseline_csv
    )

    baseline_df["country"] = (
        baseline_df["graph_name"]
        .astype(str)
    )

    missing_metrics = [
        metric
        for metric in METRICS
        if metric not in baseline_df.columns
    ]

    if missing_metrics:
        raise ValueError(
            "Baseline CSV is missing the "
            "following columns:\n"
            + "\n".join(
                missing_metrics
            )
        )

    return baseline_df


def plot_country_distributions(
    df,
    baseline_df,
    output_folder,
    experiment_name,
    bins="auto",
):
    """
    Make one 3 x 3 histogram figure per country.

    Histograms show flipped-network distributions.
    Dashed vertical lines show the original-network values.
    """
    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    countries = sorted(
        df["country"].unique()
    )

    for country in countries:

        country_df = df[
            df["country"] == country
        ]

        baseline_row = baseline_df[
            baseline_df["country"]
            == country
        ]

        if baseline_row.empty:
            print(
                f"No baseline found for "
                f"{country}; skipping."
            )
            continue

        baseline_row = (
            baseline_row.iloc[0]
        )

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(18, 13),
        )

        axes = axes.flatten()

        for ax, (
            metric,
            label,
        ) in zip(
            axes,
            METRICS.items(),
        ):

            values = pd.to_numeric(
                country_df[metric],
                errors="coerce",
            ).dropna()

            baseline_value = pd.to_numeric(
                pd.Series(
                    [
                        baseline_row[
                            metric
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if len(values) > 0:

                ax.hist(
                    values,
                    bins=bins,
                    edgecolor="black",
                    alpha=0.75,
                )

            if pd.notna(
                baseline_value
            ):

                ax.axvline(
                    baseline_value,
                    linestyle="--",
                    linewidth=2,
                    label=(
                        "Original network"
                    ),
                )

            ax.set_title(
                label,
                fontsize=10,
            )

            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

            if pd.notna(
                baseline_value
            ):
                ax.legend(
                    fontsize=8
                )

        fig.suptitle(
            f"{country} — "
            f"{experiment_name}",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            output_folder
            / (
                f"{country}_"
                "triangle_distributions.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_one_dimension_country_points(
    df,
    baseline_df,
    output_folder,
):
    """
    Make one 3 x 3 figure per country for one-dimension flips.

    Each point is one flipped dimension.
    The dashed horizontal line is the original-network value.
    """
    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    countries = sorted(
        df["country"].unique()
    )

    for country in countries:

        country_df = (
            df[
                df["country"] == country
            ]
            .copy()
            .sort_values("sample")
        )

        baseline_row = baseline_df[
            baseline_df["country"]
            == country
        ]

        if baseline_row.empty:
            print(
                f"No baseline found for "
                f"{country}; skipping."
            )
            continue

        baseline_row = (
            baseline_row.iloc[0]
        )

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(22, 14),
        )

        axes = axes.flatten()

        for ax, (
            metric,
            label,
        ) in zip(
            axes,
            METRICS.items(),
        ):

            values = pd.to_numeric(
                country_df[metric],
                errors="coerce",
            )

            dimensions = (
                country_df["sample"]
                .astype(str)
            )

            baseline_value = pd.to_numeric(
                pd.Series(
                    [
                        baseline_row[
                            metric
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            x = np.arange(
                len(country_df)
            )

            valid = values.notna()

            ax.scatter(
                x[valid],
                values[valid],
                s=45,
                zorder=3,
            )

            if pd.notna(
                baseline_value
            ):

                ax.axhline(
                    baseline_value,
                    linestyle="--",
                    linewidth=1.5,
                    label=(
                        "Original network"
                    ),
                    zorder=2,
                )

            ax.set_xticks(x)

            ax.set_xticklabels(
                dimensions,
                rotation=60,
                ha="right",
                fontsize=7,
            )

            ax.set_title(
                label,
                fontsize=10,
            )

            ax.set_xlabel(
                "Flipped dimension"
            )

            ax.set_ylabel(
                "Value"
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            if pd.notna(
                baseline_value
            ):
                ax.legend(
                    fontsize=8
                )

        fig.suptitle(
            f"{country} — "
            "One-dimension flips",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            output_folder
            / (
                f"{country}_"
                "triangle_metrics_"
                "by_dimension.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def save_country_metric_summary(
    df,
    output_csv_path,
):
    """
    Save mean and standard deviation of all triangle metrics
    across flipped networks, grouped by country.

    The summary is agnostic to which variable(s) were flipped.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_columns = list(METRICS.keys())

    summary = (
        df
        .groupby("country")[metric_columns]
        .agg(["mean", "std"])
    )

    # Flatten MultiIndex columns:
    # ratio_three_negative_mean,
    # ratio_three_negative_std, etc.
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        output_csv_path,
        index=False,
        float_format="%.3f",
    )

    return summary


def run_triangle_robustness_analysis(
    networks_root,
    baseline_csv,
    output_root,
):
    networks_root = Path(
        networks_root
    )

    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_df = (
        load_baseline_metrics(
            baseline_csv
        )
    )

    # ==================================================
    # 1. ONE-DIMENSION FLIPS
    # ==================================================

    one_dimension_paths = sorted(
        (
            networks_root
            / "flipped"
        ).glob(
            "*_flipped.graphml"
        )
    )

    one_dimension_df = (
        analyse_flipped_networks(
            one_dimension_paths,
            experiment=(
                "one_dimension"
            ),
        )
    )

    one_dimension_df.to_csv(
        output_root
        / (
            "one_dimension_"
            "triangle_metrics.csv"
        ),
        index=False,
        float_format="%.3f",
    )

    save_country_metric_summary(
        one_dimension_df,
        output_root
        / "one_dimension_triangle_metrics_summary.csv",
    )

    # Histogram distributions across
    # the 20 one-dimension flips.
    plot_country_distributions(
        df=one_dimension_df,
        baseline_df=baseline_df,
        output_folder=(
            output_root
            / "one_dimension"
            / "histograms"
        ),
        experiment_name=(
            "One-dimension flips"
        ),
    )

    # Individual points for each
    # flipped dimension.
    plot_one_dimension_country_points(
        df=one_dimension_df,
        baseline_df=baseline_df,
        output_folder=(
            output_root
            / "one_dimension"
            / "specific"
        ),
    )

    # ==================================================
    # 2. FIVE-DIMENSION RANDOM FLIPS
    # ==================================================

    five_dimension_paths = sorted(
        (
            networks_root
            / "random_flip"
            / "networks"
        ).glob(
            "*/*.graphml"
        )
    )

    five_dimension_df = (
        analyse_flipped_networks(
            five_dimension_paths,
            experiment=(
                "five_dimension"
            ),
        )
    )

    five_dimension_df.to_csv(
        output_root
        / (
            "five_dimension_"
            "triangle_metrics.csv"
        ),
        index=False,
        float_format="%.3f",
    )

    save_country_metric_summary(
        five_dimension_df,
        output_root
        / "five_dimension_triangle_metrics_summary.csv",
    )

    plot_country_distributions(
        df=five_dimension_df,
        baseline_df=baseline_df,
        output_folder=(
            output_root
            / "five_dimension"
        ),
        experiment_name=(
            "Five-dimension random flips"
        ),
    )

    return (
        one_dimension_df,
        five_dimension_df,
    )


if __name__ == "__main__":

    run_triangle_robustness_analysis(
        networks_root=(
            "../output/networks"
        ),
        baseline_csv=(
            "../output/"
            "signed_triangle_counts_expanded_2.csv"
        ),
        output_root=(
            "../plots/"
            "variable_coding_robustness_2"
        ),
    )