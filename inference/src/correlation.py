from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def retrieve_group(
    csv_path,
    filters=None,
):
    """
    Load the dataset and return the filtered respondents.

    filters example:
    {
        "age": {"range": [15, 34]},
        "country": {"value": "GB"}
    }
    """

    df = pd.read_csv(csv_path)

    if filters is None:
        filters = {}

    for column, condition in filters.items():

        if "value" in condition:
            df = df[df[column] == condition["value"]]

        if "range" in condition:
            low, high = condition["range"]

            df = df[
                (df[column] >= low)
                & (df[column] <= high)
            ]

    return df


def generate_correlation_matrix(
    df,
    question_start_col=None,
    question_end_col=None,
    question_cols=None,
    method="kendall",
):
    """
    Compute a correlation matrix from a dataframe.
    """

    if question_cols is not None:
        X = df[question_cols]

    else:
        X = df.iloc[
            :,
            question_start_col:question_end_col
        ]

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.dropna()

    corr = X.corr(method=method)

    return corr


import numpy as np
import pandas as pd


def threshold_signed_adjacency(
    A,
    top_percent=20,
    keep_diagonal=False,
):
    """
    Keep the strongest positive and strongest negative edges separately.

    Parameters
    ----------
    A : pandas.DataFrame or numpy.ndarray
        Symmetric adjacency matrix.
    top_percent : float
        Percentage of positive and negative edges to keep.
    keep_diagonal : bool
        Whether to keep the diagonal.

    Returns
    -------
    A_thr : pandas.DataFrame
        Thresholded adjacency matrix.
    pos_threshold : float
        Positive cutoff.
    neg_threshold : float
        Negative cutoff.
    """

    # Preserve labels if present
    if isinstance(A, pd.DataFrame):
        index = A.index
        columns = A.columns
        A = A.to_numpy()
    else:
        index = None
        columns = None
        A = np.asarray(A, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric.")

    if not (0 < top_percent <= 100):
        raise ValueError(
            "top_percent must be between 0 and 100."
        )

    upper_mask = np.triu(
        np.ones_like(A, dtype=bool),
        k=1,
    )

    edge_values = A[upper_mask]

    positive_edges = edge_values[edge_values > 0]
    negative_edges = edge_values[edge_values < 0]

    A_thr = np.zeros_like(A)

    pos_threshold = None
    neg_threshold = None

    # Positive edges
    if len(positive_edges) > 0:
        pos_threshold = np.percentile(
            positive_edges,
            100 - top_percent,
        )

        pos_keep = (A >= pos_threshold) & (A > 0)

        A_thr[pos_keep] = A[pos_keep]

    # Negative edges
    if len(negative_edges) > 0:
        neg_threshold = np.percentile(
            negative_edges,
            top_percent,
        )

        neg_keep = (A <= neg_threshold) & (A < 0)

        A_thr[neg_keep] = A[neg_keep]

    if not keep_diagonal:
        np.fill_diagonal(A_thr, 0)

    A_thr = np.triu(A_thr, 1)
    A_thr = A_thr + A_thr.T

    # Convert back to DataFrame
    if index is not None:
        A_thr = pd.DataFrame(
            A_thr,
            index=index,
            columns=columns,
        )

    return A_thr, pos_threshold, neg_threshold


def plot_correlation_matrix(
    corr,
    output_path=None,
    title="Correlation Matrix",
    cmap="RdBu",
):
    """
    Plot and optionally save a correlation matrix heatmap.
    """

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_correlation_network(
    corr,
    threshold=0,
    title=None,
    ax=None,
    figsize=(14, 14),
    layout_k=1.2,
    layout_iterations=500,
    seed=42,
    output_path=None,
):
    """
    Draw a correlation network with matplotlib.

    Nodes are questions/columns. Edges are coloured blue for positive
    correlations and red for negative correlations. Edge thickness is
    proportional to absolute correlation.

    Parameters
    ----------
    corr : pandas.DataFrame
        Correlation matrix.
    threshold : float
        Minimum absolute correlation needed to draw an edge.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Created if not provided.
    figsize : tuple
        Figure size if ax is not provided.
    layout_k : float
        Spacing parameter for spring layout.
    layout_iterations : int
        Number of layout iterations.
    seed : int
        Random seed for reproducible layout.
    output_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """

    adjacency = corr.copy()

    # Remove self-correlations.
    adjacency = adjacency.mask(
        np.eye(len(adjacency), dtype=bool),
        0,
    )

    # Remove weak correlations.
    adjacency[np.abs(adjacency) < threshold] = 0

    # Build NetworkX graph.
    G = nx.from_pandas_adjacency(adjacency)

    # Remove zero-weight edges.
    G.remove_edges_from([
        (u, v)
        for u, v, w in G.edges(data="weight")
        if w == 0
    ])

    # Remove isolated nodes for a cleaner plot.
    G.remove_nodes_from(list(nx.isolates(G)))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.set_aspect("equal")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.set_title("No edges above threshold", fontsize=11)
        return ax

    # Spring layout is the NetworkX equivalent of a force-directed layout.
    pos = nx.spring_layout(
        G,
        seed=seed,
        k=layout_k,
        iterations=layout_iterations,
        weight="weight",
    )

    coords = np.array([pos[node] for node in G.nodes()])
    x = coords[:, 0]
    y = coords[:, 1]

    node_to_i = {
        node: i
        for i, node in enumerate(G.nodes())
    }

    # Draw edges manually for better control.
    for u, v, data in G.edges(data=True):
        weight = data["weight"]

        color = "#d62728" if weight < 0 else "#1f77b4"
        lw = np.clip(abs(weight) * 8, 0.4, 4.0)

        u_i = node_to_i[u]
        v_i = node_to_i[v]

        ax.plot(
            [x[u_i], x[v_i]],
            [y[u_i], y[v_i]],
            color=color,
            linewidth=lw,
            alpha=0.6,
            zorder=1,
            solid_capstyle="round",
        )

    # Draw nodes.
    ax.scatter(
        x,
        y,
        s=300,
        color="white",
        edgecolors="#333333",
        linewidths=1.5,
        zorder=2,
    )

    # Draw labels slightly outside the node cloud.
    cx, cy = x.mean(), y.mean()
    span = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
    )

    for node in G.nodes():
        i = node_to_i[node]

        dx = x[i] - cx
        dy = y[i] - cy

        norm = max(np.hypot(dx, dy), 1e-6)
        offset = 0.03 * span

        lx = x[i] + dx / norm * offset
        ly = y[i] + dy / norm * offset

        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        ax.text(
            lx,
            ly,
            str(node),
            fontsize=9,
            ha=ha,
            va=va,
            zorder=3,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.7,
            ),
        )

    if title is None:
        title = (
            "Correlation network\n"
            "Blue = positive  |  Red = negative"
        )

    ax.set_title(title, fontsize=11)

    if output_path is not None:
        ax.figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return ax


def save_correlation_matrix(corr, output_path):
    """
    Save a correlation matrix.

    Supported formats:
    - .csv
    - .parquet
    - .pkl
    - .npy
    """

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        corr.to_csv(output_path)

    elif suffix == ".parquet":
        corr.to_parquet(output_path)

    elif suffix in [".pkl", ".pickle"]:
        corr.to_pickle(output_path)

    elif suffix == ".npy":
        np.save(output_path, corr.to_numpy())

    else:
        raise ValueError(
            "Unsupported file format. Use .csv, .parquet, .pkl, or .npy"
        )