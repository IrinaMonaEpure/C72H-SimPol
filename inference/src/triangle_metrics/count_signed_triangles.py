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
    Calculate weighted metrics for one triangle.

    Parameters
    ----------
    weights : iterable of float
        The three signed edge weights of a triangle.

    Returns
    -------
    tuple[float, float]
        absolute_weight_sum:
            Sum of the three absolute edge weights:
            |w1| + |w2| + |w3|.

        onnela_intensity:
            Geometric mean of the three absolute edge weights:
            (|w1 * w2 * w3|)^(1/3).
    """
    absolute_weights = np.abs(
        np.asarray(weights, dtype=float)
    )

    absolute_weight_sum = float(
        np.sum(absolute_weights)
    )

    onnela_intensity = float(
        np.prod(absolute_weights) ** (1.0 / 3.0)
    )

    return absolute_weight_sum, onnela_intensity


def count_signed_triangles(
    G,
    weight_attribute="weight",
):
    """
    Count signed triangles and aggregate their weighted metrics.

    Imbalanced triangles are defined as:
    - three negative edges;
    - two positive edges and one negative edge.

    For each selected triangle type, the function calculates:
    - number of triangles;
    - sum of absolute edge weights;
    - sum of Onnela intensities.

    Returns
    -------
    dict
        Triangle counts and weighted sums.
    """
    if G.is_directed():
        raise ValueError(
            "The graph is directed. "
            "This function expects undirected graphs."
        )

    total_triangles = 0

    three_negative_edges = 0
    two_positive_one_negative = 0

    three_negative_absolute_weight_sum = 0.0
    two_positive_one_negative_absolute_weight_sum = 0.0

    three_negative_onnela_intensity_sum = 0.0
    two_positive_one_negative_onnela_intensity_sum = 0.0

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
                ) = calculate_triangle_weight_metrics(
                    weights
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

    return {
        "total_triangles": total_triangles,

        "three_negative_edges": (
            three_negative_edges
        ),

        "two_positive_one_negative": (
            two_positive_one_negative
        ),

        "imbalanced_triangles": (
            imbalanced_triangles
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


def analyse_graphml_folder(
    input_folder,
    output_csv_path,
    pattern="*.graphml",
    weight_attribute="weight",
):
    """
    Analyse signed triangles in all GraphML files in a folder.

    The CSV contains:
    - total triangle count;
    - counts and ratios of:
        * three-negative triangles;
        * two-positive/one-negative triangles;
        * all imbalanced triangles;
    - sums of absolute edge weights for each imbalanced type
      and for all imbalanced triangles combined;
    - sums of Onnela intensities for each imbalanced type
      and for all imbalanced triangles combined.

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

                ratio_imbalanced = (
                    metrics["imbalanced_triangles"]
                    / total_triangles
                )

            else:
                ratio_three_negative = 0.0
                ratio_two_positive_one_negative = 0.0
                ratio_imbalanced = 0.0

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

                    "three_negative_edges": (
                        metrics[
                            "three_negative_edges"
                        ]
                    ),

                    "ratio_three_negative": (
                        ratio_three_negative
                    ),

                    "two_positive_one_negative": (
                        metrics[
                            "two_positive_one_negative"
                        ]
                    ),

                    "ratio_two_positive_one_negative": (
                        ratio_two_positive_one_negative
                    ),

                    "imbalanced_triangles": (
                        metrics[
                            "imbalanced_triangles"
                        ]
                    ),

                    "ratio_imbalanced": (
                        ratio_imbalanced
                    ),

                    "three_negative_absolute_weight_sum": (
                        metrics[
                            "three_negative_absolute_weight_sum"
                        ]
                    ),

                    "two_positive_one_negative_absolute_weight_sum": (
                        metrics[
                            "two_positive_one_negative_absolute_weight_sum"
                        ]
                    ),

                    "imbalanced_absolute_weight_sum": (
                        metrics[
                            "imbalanced_absolute_weight_sum"
                        ]
                    ),

                    "three_negative_onnela_intensity_sum": (
                        metrics[
                            "three_negative_onnela_intensity_sum"
                        ]
                    ),

                    "two_positive_one_negative_onnela_intensity_sum": (
                        metrics[
                            "two_positive_one_negative_onnela_intensity_sum"
                        ]
                    ),

                    "imbalanced_onnela_intensity_sum": (
                        metrics[
                            "imbalanced_onnela_intensity_sum"
                        ]
                    ),
                }
            )

        except Exception as exc:
            tqdm.write(
                f"Failed on {graph_path.name}: {exc}"
            )

        # Save after every graph so completed work is retained
        # if processing is interrupted.
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
            "../output/signed_triangle_counts_expanded_2.csv"
        ),
        pattern="*.graphml",
        weight_attribute="weight",
    )