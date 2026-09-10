from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_network_metrics(graph_path):
    """
    Calculate:
      1. Number of edges
      2. Ratio of imbalanced triangles
      3. Sum of absolute edge weights over the entire network

    Imbalanced triangles are:
      - 3 negative edges
      - 2 positive edges + 1 negative edge
    """

    G = nx.read_graphml(graph_path)

    # ----------------------------------------------------------
    # Number of edges
    # ----------------------------------------------------------
    number_of_edges = G.number_of_edges()

    # ----------------------------------------------------------
    # Sum of absolute edge weights over the WHOLE network
    # ----------------------------------------------------------
    edge_weight_sum = sum(
        abs(float(data["weight"]))
        for _, _, data in G.edges(data=True)
    )

    # If you instead want the signed sum:
    #
    # edge_weight_sum = sum(
    #     float(data["weight"])
    #     for _, _, data in G.edges(data=True)
    # )

    # ----------------------------------------------------------
    # Count triangles
    # ----------------------------------------------------------
    total_triangles = 0
    imbalanced_triangles = 0

    nodes = list(G.nodes())

    node_order = {
        node: i
        for i, node in enumerate(nodes)
    }

    # Keep only "forward" neighbors so that every triangle
    # is counted exactly once.
    forward_neighbors = {}

    for u in nodes:
        forward_neighbors[u] = {
            v
            for v in G.neighbors(u)
            if node_order[v] > node_order[u]
        }

    for u in nodes:
        for v in forward_neighbors[u]:

            common_neighbors = (
                forward_neighbors[u]
                & forward_neighbors[v]
            )

            for w in common_neighbors:

                weights = [
                    float(G[u][v]["weight"]),
                    float(G[u][w]["weight"]),
                    float(G[v][w]["weight"]),
                ]

                total_triangles += 1

                n_negative = sum(
                    weight < 0
                    for weight in weights
                )

                # Structurally imbalanced triangles:
                #
                # --- : 3 negative edges
                # ++- : 1 negative edge
                if n_negative in (1, 3):
                    imbalanced_triangles += 1

    ratio_imbalanced = (
        imbalanced_triangles / total_triangles
        if total_triangles > 0
        else np.nan
    )

    return {
        "number_of_edges": number_of_edges,
        "total_triangles": total_triangles,
        "imbalanced_triangles": imbalanced_triangles,
        "ratio_imbalanced": ratio_imbalanced,
        "edge_weight_sum": edge_weight_sum,
    }


def analyse_original_and_one_variable_flips(
    baseline_folder="inference/output/networks",
    flipped_folder="inference/output/networks/flipped",
    output_individual_csv=(
        "inference/output/"
        "original_and_one_variable_flipped_metrics.csv"
    ),
    output_mean_csv=(
        "inference/output/"
        "original_and_one_variable_flipped_metrics_mean.csv"
    ),
):
    baseline_folder = Path(baseline_folder)
    flipped_folder = Path(flipped_folder)

    output_individual_csv = Path(output_individual_csv)
    output_mean_csv = Path(output_mean_csv)

    rows = []

    # ==========================================================
    # Original networks
    # ==========================================================
    baseline_files = sorted(
        baseline_folder.glob("*.graphml")
    )

    print(
        f"Processing {len(baseline_files)} original networks..."
    )

    for graph_path in tqdm(
        baseline_files,
        desc="Original",
    ):
        country = graph_path.stem

        metrics = calculate_network_metrics(
            graph_path
        )

        rows.append(
            {
                "country": country,
                "network_type": "original",
                "flipped_dimension": None,
                "file_name": graph_path.name,
                **metrics,
            }
        )

    # ==========================================================
    # One-variable flipped networks
    #
    # Expected filename format:
    #
    # GB_anti_immigration_flipped.graphml
    # ==========================================================
    flipped_files = sorted(
        flipped_folder.glob("*.graphml")
    )

    print(
        f"\nProcessing {len(flipped_files)} "
        "one-variable flipped networks..."
    )

    for graph_path in tqdm(
        flipped_files,
        desc="One-variable flips",
    ):
        stem = graph_path.stem

        # Example:
        #
        # GB_anti_immigration_flipped
        #
        # country = GB
        country = stem.split("_", 1)[0]

        # Everything after "GB_"
        flipped_dimension = stem[
            len(country) + 1:
        ]

        # Remove trailing "_flipped"
        if flipped_dimension.endswith("_flipped"):
            flipped_dimension = (
                flipped_dimension[
                    :-len("_flipped")
                ]
            )

        metrics = calculate_network_metrics(
            graph_path
        )

        rows.append(
            {
                "country": country,
                "network_type": "one_variable_flipped",
                "flipped_dimension": flipped_dimension,
                "file_name": graph_path.name,
                **metrics,
            }
        )

    # ==========================================================
    # Individual-network results
    # ==========================================================
    individual_df = pd.DataFrame(rows)

    individual_df = individual_df.sort_values(
        [
            "country",
            "network_type",
            "flipped_dimension",
        ],
        na_position="first",
    ).reset_index(drop=True)

    output_individual_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_df.to_csv(
        output_individual_csv,
        index=False,
        float_format="%.6f",
    )

    # ==========================================================
    # Mean per country
    #
    # Includes:
    #   1 original network
    #   + 20 one-variable flipped networks
    #
    # Normally n_networks = 21.
    # ==========================================================
    mean_df = (
        individual_df
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            number_of_edges_mean=(
                "number_of_edges",
                "mean",
            ),
            ratio_imbalanced_mean=(
                "ratio_imbalanced",
                "mean",
            ),
            edge_weight_sum_mean=(
                "edge_weight_sum",
                "mean",
            ),
            n_networks=(
                "file_name",
                "count",
            ),
        )
    )

    output_mean_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean_df.to_csv(
        output_mean_csv,
        index=False,
        float_format="%.6f",
    )

    # ==========================================================
    # Print results
    # ==========================================================
    print(
        "\nMean metrics per country "
        "(original + all one-variable flips):"
    )

    print(
        mean_df.to_string(
            index=False
        )
    )

    # Warn if a country does not have the expected
    # 1 original + 20 flipped = 21 networks.
    unexpected_counts = mean_df[
        mean_df["n_networks"] != 21
    ]

    if not unexpected_counts.empty:
        print(
            "\nWARNING: Some countries do not have "
            "21 networks:"
        )

        print(
            unexpected_counts[
                [
                    "country",
                    "n_networks",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        f"\nIndividual results saved to:\n"
        f"{output_individual_csv}"
    )

    print(
        f"\nCountry means saved to:\n"
        f"{output_mean_csv}"
    )

    return individual_df, mean_df


if __name__ == "__main__":
    individual_df, mean_df = (
        analyse_original_and_one_variable_flips()
    )