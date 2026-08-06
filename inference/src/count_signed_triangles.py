from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_edge_weight(G, u, v, weight_attribute="weight"):
    """
    Return an edge's weight as a float.

    For a MultiGraph, this requires exactly one edge between u and v.
    """
    edge_data = G.get_edge_data(u, v)

    if edge_data is None:
        raise KeyError(f"No edge found between {u!r} and {v!r}.")

    if G.is_multigraph():
        if len(edge_data) != 1:
            raise ValueError(
                f"Multiple edges found between {u!r} and {v!r}. "
                "Specify how parallel-edge weights should be combined."
            )

        edge_data = next(iter(edge_data.values()))

    if weight_attribute not in edge_data:
        raise KeyError(
            f"Edge ({u!r}, {v!r}) has no "
            f"{weight_attribute!r} attribute."
        )

    try:
        return float(edge_data[weight_attribute])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Edge ({u!r}, {v!r}) has a non-numeric "
            f"{weight_attribute!r} value: "
            f"{edge_data[weight_attribute]!r}"
        ) from exc


def calculate_triangle_weight_metrics(weights):
    """
    Calculate non-negative weighted metrics for one triangle.

    Parameters
    ----------
    weights : iterable of float
        The three signed edge weights of a triangle.

    Returns
    -------
    tuple[float, float]
        absolute_mean:
            Arithmetic mean of the three absolute edge weights.

        onnela_geometric_mean:
            Geometric mean of the three absolute edge weights.
    """
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


def count_signed_triangles(
    G,
    weight_attribute="weight",
):
    """
    Count selected signed triangle types and calculate average
    weighted metrics.

    Metrics are calculated for:

    - all triangles;
    - triangles with three negative edges;
    - triangles with two positive edges and one negative edge.

    Returns
    -------
    dict
        Triangle counts and average weight metrics.
    """
    if G.is_directed():
        raise ValueError(
            "The graph is directed. "
            "This function expects undirected graphs."
        )

    total_triangles = 0

    three_negative_edges = 0
    two_positive_one_negative = 0

    overall_absolute_mean_sum = 0.0
    overall_onnela_geometric_mean_sum = 0.0

    three_negative_absolute_mean_sum = 0.0
    three_negative_onnela_geometric_mean_sum = 0.0

    two_positive_one_negative_absolute_mean_sum = 0.0
    two_positive_one_negative_onnela_geometric_mean_sum = 0.0

    # Ensure every triangle is visited exactly once.
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
                        weight_attribute=weight_attribute,
                    ),
                    get_edge_weight(
                        G,
                        u,
                        w,
                        weight_attribute=weight_attribute,
                    ),
                    get_edge_weight(
                        G,
                        v,
                        w,
                        weight_attribute=weight_attribute,
                    ),
                ]

                absolute_mean, onnela_geometric_mean = (
                    calculate_triangle_weight_metrics(
                        weights
                    )
                )

                total_triangles += 1

                overall_absolute_mean_sum += (
                    absolute_mean
                )
                overall_onnela_geometric_mean_sum += (
                    onnela_geometric_mean
                )

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

                    three_negative_absolute_mean_sum += (
                        absolute_mean
                    )
                    three_negative_onnela_geometric_mean_sum += (
                        onnela_geometric_mean
                    )

                elif (
                    positive_count == 2
                    and negative_count == 1
                ):
                    two_positive_one_negative += 1

                    two_positive_one_negative_absolute_mean_sum += (
                        absolute_mean
                    )
                    two_positive_one_negative_onnela_geometric_mean_sum += (
                        onnela_geometric_mean
                    )

    if total_triangles > 0:
        overall_mean_absolute_weight = (
            overall_absolute_mean_sum
            / total_triangles
        )

        overall_mean_onnela_geometric_mean = (
            overall_onnela_geometric_mean_sum
            / total_triangles
        )
    else:
        overall_mean_absolute_weight = np.nan
        overall_mean_onnela_geometric_mean = np.nan

    if three_negative_edges > 0:
        three_negative_mean_absolute_weight = (
            three_negative_absolute_mean_sum
            / three_negative_edges
        )

        three_negative_mean_onnela_geometric_mean = (
            three_negative_onnela_geometric_mean_sum
            / three_negative_edges
        )
    else:
        three_negative_mean_absolute_weight = np.nan
        three_negative_mean_onnela_geometric_mean = np.nan

    if two_positive_one_negative > 0:
        two_positive_one_negative_mean_absolute_weight = (
            two_positive_one_negative_absolute_mean_sum
            / two_positive_one_negative
        )

        two_positive_one_negative_mean_onnela_geometric_mean = (
            two_positive_one_negative_onnela_geometric_mean_sum
            / two_positive_one_negative
        )
    else:
        two_positive_one_negative_mean_absolute_weight = np.nan
        two_positive_one_negative_mean_onnela_geometric_mean = np.nan

    return {
        "total_triangles": total_triangles,
        "overall_mean_absolute_weight": (
            overall_mean_absolute_weight
        ),
        "overall_mean_onnela_geometric_mean": (
            overall_mean_onnela_geometric_mean
        ),
        "three_negative_edges": (
            three_negative_edges
        ),
        "three_negative_mean_absolute_weight": (
            three_negative_mean_absolute_weight
        ),
        "three_negative_mean_onnela_geometric_mean": (
            three_negative_mean_onnela_geometric_mean
        ),
        "two_positive_one_negative": (
            two_positive_one_negative
        ),
        "two_positive_one_negative_mean_absolute_weight": (
            two_positive_one_negative_mean_absolute_weight
        ),
        "two_positive_one_negative_mean_onnela_geometric_mean": (
            two_positive_one_negative_mean_onnela_geometric_mean
        ),
    }


def analyse_graphml_folder(
    input_folder,
    output_csv_path,
    pattern="*.graphml",
    weight_attribute="weight",
):
    """
    Analyse signed triangles in all GraphML files in a folder.

    The CSV contains:

    - total triangle counts;
    - overall average triangle weight metrics;
    - counts and ratios for selected signed triangle types;
    - average weight metrics for each selected signed type.

    Returns
    -------
    pandas.DataFrame
        One row per successfully processed graph.
    """
    input_folder = Path(input_folder)
    output_csv_path = Path(output_csv_path)

    graph_paths = sorted(
        input_folder.glob(pattern)
    )

    if not graph_paths:
        raise ValueError(
            f"No files matching {pattern!r} found in "
            f"{input_folder.resolve()}"
        )

    output_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    progress = tqdm(
        graph_paths,
        total=len(graph_paths),
        desc="Processing graphs",
        unit="graph",
    )

    for graph_path in progress:
        progress.set_postfix_str(
            graph_path.name
        )

        try:
            G = nx.read_graphml(
                graph_path
            )

            metrics = count_signed_triangles(
                G,
                weight_attribute=weight_attribute,
            )

            total_triangles = (
                metrics["total_triangles"]
            )

            if total_triangles > 0:
                ratio_three_negative = (
                    metrics["three_negative_edges"]
                    / total_triangles
                )

                ratio_two_positive_one_negative = (
                    metrics[
                        "two_positive_one_negative"
                    ]
                    / total_triangles
                )
            else:
                ratio_three_negative = 0.0
                ratio_two_positive_one_negative = 0.0

            rows.append(
                {
                    "graph_name": graph_path.stem,
                    "file_name": graph_path.name,
                    "number_of_nodes": (
                        G.number_of_nodes()
                    ),
                    "number_of_edges": (
                        G.number_of_edges()
                    ),
                    "total_triangles": (
                        total_triangles
                    ),
                    "overall_mean_absolute_weight": (
                        metrics[
                            "overall_mean_absolute_weight"
                        ]
                    ),
                    "overall_mean_onnela_geometric_mean": (
                        metrics[
                            "overall_mean_onnela_geometric_mean"
                        ]
                    ),
                    "three_negative_edges": (
                        metrics[
                            "three_negative_edges"
                        ]
                    ),
                    "ratio_three_negative": (
                        ratio_three_negative
                    ),
                    "three_negative_mean_absolute_weight": (
                        metrics[
                            "three_negative_mean_absolute_weight"
                        ]
                    ),
                    "three_negative_mean_onnela_geometric_mean": (
                        metrics[
                            "three_negative_mean_onnela_geometric_mean"
                        ]
                    ),
                    "two_positive_one_negative": (
                        metrics[
                            "two_positive_one_negative"
                        ]
                    ),
                    "ratio_two_positive_one_negative": (
                        ratio_two_positive_one_negative
                    ),
                    "two_positive_one_negative_mean_absolute_weight": (
                        metrics[
                            "two_positive_one_negative_mean_absolute_weight"
                        ]
                    ),
                    "two_positive_one_negative_mean_onnela_geometric_mean": (
                        metrics[
                            "two_positive_one_negative_mean_onnela_geometric_mean"
                        ]
                    ),
                }
            )

        except Exception as exc:
            tqdm.write(
                f"Failed on {graph_path.name}: {exc}"
            )

        pd.DataFrame(rows).to_csv(
            output_csv_path,
            index=False,
            float_format="%.3f",
        )

    result = pd.DataFrame(rows)

    print(
        f"Processed {len(result)} of "
        f"{len(graph_paths)} graphs."
    )
    print(
        f"Results saved to: "
        f"{output_csv_path.resolve()}"
    )

    return result


if __name__ == "__main__":
    analyse_graphml_folder(
        input_folder="../output/networks",
        output_csv_path=(
            "../output/signed_triangle_counts.csv"
        ),
        pattern="*.graphml",
        weight_attribute="weight",
    )