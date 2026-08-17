from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


METRICS = {
    "ratio_three_negative": "Ratio of 3-negative triangles",
    "ratio_two_positive_one_negative": "Ratio of 2-positive, 1-negative triangles",
    "overall_mean_absolute_weight": "Mean absolute weight — all triangles",
    "three_negative_mean_absolute_weight": "Mean absolute weight — 3-negative",
    "two_positive_one_negative_mean_absolute_weight": (
        "Mean absolute weight — 2-positive, 1-negative"
    ),
    "overall_mean_onnela_geometric_mean": "Onnela intensity — all triangles",
    "three_negative_mean_onnela_geometric_mean": (
        "Onnela intensity — 3-negative"
    ),
    "two_positive_one_negative_mean_onnela_geometric_mean": (
        "Onnela intensity — 2-positive, 1-negative"
    ),
}


def get_edge_weight(G, u, v, weight_attribute="weight"):
    edge_data = G.get_edge_data(u, v)

    if edge_data is None:
        raise KeyError(f"No edge found between {u!r} and {v!r}.")

    if G.is_multigraph():
        if len(edge_data) != 1:
            raise ValueError(
                f"Multiple edges found between {u!r} and {v!r}."
            )
        edge_data = next(iter(edge_data.values()))

    if weight_attribute not in edge_data:
        raise KeyError(
            f"Edge ({u!r}, {v!r}) has no "
            f"{weight_attribute!r} attribute."
        )

    return float(edge_data[weight_attribute])


def calculate_triangle_weight_metrics(weights):
    absolute_weights = np.abs(
        np.asarray(weights, dtype=float)
    )

    absolute_mean = float(
        np.mean(absolute_weights)
    )

    onnela_geometric_mean = float(
        np.prod(absolute_weights) ** (1.0 / 3.0)
    )

    return absolute_mean, onnela_geometric_mean


def calculate_graph_triangle_metrics(
    G,
    weight_attribute="weight",
):
    if G.is_directed():
        raise ValueError(
            "Expected an undirected graph."
        )

    total_triangles = 0

    three_negative_edges = 0
    two_positive_one_negative = 0

    overall_absolute_sum = 0.0
    overall_onnela_sum = 0.0

    three_negative_absolute_sum = 0.0
    three_negative_onnela_sum = 0.0

    two_positive_one_negative_absolute_sum = 0.0
    two_positive_one_negative_onnela_sum = 0.0

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

                absolute_mean, onnela = (
                    calculate_triangle_weight_metrics(
                        weights
                    )
                )

                total_triangles += 1

                overall_absolute_sum += absolute_mean
                overall_onnela_sum += onnela

                negative_count = sum(
                    weight < 0
                    for weight in weights
                )

                positive_count = sum(
                    weight > 0
                    for weight in weights
                )

                if negative_count == 3:

                    three_negative_edges += 1

                    three_negative_absolute_sum += (
                        absolute_mean
                    )

                    three_negative_onnela_sum += (
                        onnela
                    )

                elif (
                    positive_count == 2
                    and negative_count == 1
                ):

                    two_positive_one_negative += 1

                    two_positive_one_negative_absolute_sum += (
                        absolute_mean
                    )

                    two_positive_one_negative_onnela_sum += (
                        onnela
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

        overall_mean_absolute_weight = (
            overall_absolute_sum
            / total_triangles
        )

        overall_mean_onnela_geometric_mean = (
            overall_onnela_sum
            / total_triangles
        )

    else:

        ratio_three_negative = 0.0
        ratio_two_positive_one_negative = 0.0

        overall_mean_absolute_weight = np.nan
        overall_mean_onnela_geometric_mean = np.nan

    if three_negative_edges > 0:

        three_negative_mean_absolute_weight = (
            three_negative_absolute_sum
            / three_negative_edges
        )

        three_negative_mean_onnela_geometric_mean = (
            three_negative_onnela_sum
            / three_negative_edges
        )

    else:

        three_negative_mean_absolute_weight = np.nan
        three_negative_mean_onnela_geometric_mean = np.nan

    if two_positive_one_negative > 0:

        two_positive_one_negative_mean_absolute_weight = (
            two_positive_one_negative_absolute_sum
            / two_positive_one_negative
        )

        two_positive_one_negative_mean_onnela_geometric_mean = (
            two_positive_one_negative_onnela_sum
            / two_positive_one_negative
        )

    else:

        two_positive_one_negative_mean_absolute_weight = np.nan
        two_positive_one_negative_mean_onnela_geometric_mean = np.nan

    return {
        "total_triangles": total_triangles,

        "ratio_three_negative": (
            ratio_three_negative
        ),

        "ratio_two_positive_one_negative": (
            ratio_two_positive_one_negative
        ),

        "overall_mean_absolute_weight": (
            overall_mean_absolute_weight
        ),

        "three_negative_mean_absolute_weight": (
            three_negative_mean_absolute_weight
        ),

        "two_positive_one_negative_mean_absolute_weight": (
            two_positive_one_negative_mean_absolute_weight
        ),

        "overall_mean_onnela_geometric_mean": (
            overall_mean_onnela_geometric_mean
        ),

        "three_negative_mean_onnela_geometric_mean": (
            three_negative_mean_onnela_geometric_mean
        ),

        "two_positive_one_negative_mean_onnela_geometric_mean": (
            two_positive_one_negative_mean_onnela_geometric_mean
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

        metrics = calculate_graph_triangle_metrics(
            G,
            weight_attribute=weight_attribute,
        )

        if experiment == "one_dimension":

            # Example:
            # GB_anti_immigration_flipped.graphml

            country = path.stem.split("_", 1)[0]

            suffix = "_flipped"

            name_without_country = (
                path.stem[
                    len(country) + 1:
                ]
            )

            if name_without_country.endswith(suffix):
                dimension = (
                    name_without_country[
                        :-len(suffix)
                    ]
                )
            else:
                dimension = name_without_country

            sample = dimension

        elif experiment == "five_dimension":

            # Folder itself contains country:
            #
            # .../networks/GB/GB_0.graphml

            country = path.parent.name
            sample = path.stem

        else:
            raise ValueError(
                "experiment must be "
                "'one_dimension' or 'five_dimension'"
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


def plot_country_distributions(
    df,
    baseline_df,
    output_folder,
    experiment_name,
    bins="auto",
):
    """
    Make one 2 x 4 figure for every country.

    Histograms show the distribution over flipped networks.
    Dashed vertical lines show the corresponding baseline value.
    """
    output_folder = Path(output_folder)
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
            baseline_df["country"] == country
        ]

        if baseline_row.empty:
            print(
                f"No baseline found for {country}; "
                "skipping."
            )
            continue

        baseline_row = baseline_row.iloc[0]

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(18, 9),
        )

        axes = axes.flatten()

        for ax, (metric, label) in zip(
            axes,
            METRICS.items(),
        ):

            values = pd.to_numeric(
                country_df[metric],
                errors="coerce",
            ).dropna()

            baseline_value = pd.to_numeric(
                pd.Series(
                    [baseline_row[metric]]
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

            if pd.notna(baseline_value):

                ax.axvline(
                    baseline_value,
                    linestyle="--",
                    linewidth=2,
                    label="Original network",
                )

            ax.set_title(
                label,
                fontsize=10,
            )

            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

            if pd.notna(baseline_value):
                ax.legend(
                    fontsize=8,
                )

        fig.suptitle(
            f"{country} — {experiment_name}",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            output_folder
            / f"{country}_triangle_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def load_baseline_metrics(
    baseline_csv,
):
    """
    Load the metrics calculated for the original networks.

    Expects the CSV produced by the expanded signed-triangle script.
    """
    baseline_df = pd.read_csv(
        baseline_csv
    )

    # Your existing CSV uses graph_name for the country code.
    baseline_df["country"] = (
        baseline_df["graph_name"]
        .astype(str)
    )

    return baseline_df


def plot_one_dimension_country_points(
    df,
    baseline_df,
    output_folder,
):
    """
    Make one 2 x 4 figure per country for the one-dimension flip experiment.

    Each point represents one flipped dimension.
    The dashed horizontal line shows the value for the original network.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    countries = sorted(
        df["country"].unique()
    )

    for country in countries:
        country_df = (
            df[df["country"] == country]
            .copy()
            .sort_values("sample")
        )

        baseline_row = baseline_df[
            baseline_df["country"] == country
        ]

        if baseline_row.empty:
            print(
                f"No baseline found for {country}; skipping."
            )
            continue

        baseline_row = baseline_row.iloc[0]

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(22, 10),
        )

        axes = axes.flatten()

        for ax, (metric, label) in zip(
            axes,
            METRICS.items(),
        ):
            values = pd.to_numeric(
                country_df[metric],
                errors="coerce",
            )

            dimensions = country_df["sample"].astype(str)

            baseline_value = pd.to_numeric(
                pd.Series([baseline_row[metric]]),
                errors="coerce",
            ).iloc[0]

            x = np.arange(len(country_df))

            valid = values.notna()

            ax.scatter(
                x[valid],
                values[valid],
                s=45,
                zorder=3,
            )

            if pd.notna(baseline_value):
                ax.axhline(
                    baseline_value,
                    linestyle="--",
                    linewidth=1.5,
                    label="Original network",
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

            ax.set_xlabel("Flipped dimension")
            ax.set_ylabel("Value")

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            if pd.notna(baseline_value):
                ax.legend(
                    fontsize=8,
                )

        fig.suptitle(
            f"{country} — One-dimension flips",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            output_folder
            / f"{country}_triangle_metrics_by_dimension.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def run_triangle_robustness_analysis(
    networks_root,
    baseline_csv,
    output_root,
):
    networks_root = Path(networks_root)
    output_root = Path(output_root)

    # Create output directory if it does not exist
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline_metrics(baseline_csv)

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
            experiment="one_dimension",
        )
    )

    one_dimension_df.to_csv(
        output_root
        / "one_dimension_triangle_metrics.csv",
        index=False,
        float_format="%.3f",
    )

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
            experiment="five_dimension",
        )
    )

    five_dimension_df.to_csv(
        output_root
        / "five_dimension_triangle_metrics.csv",
        index=False,
        float_format="%.3f",
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
            "signed_triangle_counts_expanded.csv"
        ),
        output_root=(
            "../plots/"
            "variable_coding_robustness"
        ),
    )