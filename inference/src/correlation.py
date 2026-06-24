from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV


def retrieve_group(csv_path, filters=None):
    df = pd.read_csv(csv_path)

    if filters is None:
        filters = {}

    for column, condition in filters.items():
        if "value" in condition:
            df = df[df[column] == condition["value"]]

        if "range" in condition:
            low, high = condition["range"]
            df = df[(df[column] >= low) & (df[column] <= high)]

    return df


def generate_correlation_matrix(
    df,
    question_start_col=None,
    question_end_col=None,
    question_cols=None,
    method="kendall",
):
    if question_cols is not None:
        X = df[question_cols]
    else:
        X = df.iloc[:, question_start_col:question_end_col]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna()

    return X.corr(method=method)


def generate_partial_correlation_matrix(
    df,
    question_start_col=None,
    question_end_col=None,
    question_cols=None,
    impute_strategy="median",
):
    if question_cols is not None:
        X = df[question_cols]
    else:
        X = df.iloc[:, question_start_col:question_end_col]

    X = X.apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)

    X_scaled = StandardScaler().fit_transform(X_imputed)

    model = GraphicalLassoCV()
    model.fit(X_scaled)

    precision = model.precision_

    partial_corr = -precision / np.sqrt(
        np.outer(np.diag(precision), np.diag(precision))
    )

    np.fill_diagonal(partial_corr, 0)

    return pd.DataFrame(
        partial_corr,
        index=X.columns,
        columns=X.columns,
    )


def threshold_signed_adjacency(A, top_percent=20, keep_diagonal=False):
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
        raise ValueError("top_percent must be between 0 and 100.")

    upper_mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    edge_values = A[upper_mask]

    positive_edges = edge_values[edge_values > 0]
    negative_edges = edge_values[edge_values < 0]

    A_thr = np.zeros_like(A)

    pos_threshold = None
    neg_threshold = None

    if len(positive_edges) > 0:
        pos_threshold = np.percentile(positive_edges, 100 - top_percent)
        pos_keep = (A >= pos_threshold) & (A > 0)
        A_thr[pos_keep] = A[pos_keep]

    if len(negative_edges) > 0:
        neg_threshold = np.percentile(negative_edges, top_percent)
        neg_keep = (A <= neg_threshold) & (A < 0)
        A_thr[neg_keep] = A[neg_keep]

    if not keep_diagonal:
        np.fill_diagonal(A_thr, 0)

    A_thr = np.triu(A_thr, 1)
    A_thr = A_thr + A_thr.T

    if index is not None:
        A_thr = pd.DataFrame(A_thr, index=index, columns=columns)

    return A_thr, pos_threshold, neg_threshold


def plot_correlation_matrix(
    corr,
    output_path=None,
    title="Correlation Matrix",
    cmap="RdBu",
    show=False,
):
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

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def plot_correlation_distribution(
    matrix,
    title="Correlation distribution",
    output_path=None,
    bins=40,
    show=False,
):
    A = matrix.to_numpy()

    upper_mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    values = A[upper_mask]
    values = values[~np.isnan(values)]

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        values,
        bins=bins,
        kde=True,
        ax=ax,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Correlation value")
    ax.set_ylabel("Count")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


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
    show=False,
):
    adjacency = corr.copy()

    adjacency = adjacency.mask(
        np.eye(len(adjacency), dtype=bool),
        0,
    )

    adjacency[np.abs(adjacency) < threshold] = 0

    G = nx.from_pandas_adjacency(adjacency)

    G.remove_edges_from([
        (u, v)
        for u, v, w in G.edges(data="weight")
        if w == 0
    ])

    G.remove_nodes_from(list(nx.isolates(G)))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.set_aspect("equal")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.set_title("No edges above threshold", fontsize=11)

        if output_path is not None:
            ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(ax.figure)

        return ax

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

    ax.scatter(
        x,
        y,
        s=300,
        color="white",
        edgecolors="#333333",
        linewidths=1.5,
        zorder=2,
    )

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
        title = "Correlation network\nBlue = positive  |  Red = negative"

    ax.set_title(title, fontsize=11)

    if output_path is not None:
        ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(ax.figure)

    return ax


def save_correlation_matrix(corr, output_path):
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


def run_correlation_pipeline(
    csv_path,
    column="cntry",
    question_start_col=14,
    question_end_col=34,
    output_dir="../outputs",
    plot_dir="../plots",
    correlation_method="kendall",
):
    output_dir = Path(output_dir)
    plot_dir = Path(plot_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    groups = sorted(df[column].dropna().unique())

    results = {}

    for group_value in groups:
        print(f"Processing {group_value}...")

        filters = {
            column: {"value": group_value}
        }

        group = retrieve_group(
            csv_path=csv_path,
            filters=filters,
        )

        if len(group) < 5:
            print(f"Skipping {group_value}: too few rows ({len(group)})")
            continue

        group_results = {}

        configs = {}

        corr = generate_correlation_matrix(
            group,
            question_start_col=question_start_col,
            question_end_col=question_end_col,
            method=correlation_method,
        )

        configs["correlation"] = corr

        try:
            part_corr = generate_partial_correlation_matrix(
                group,
                question_start_col=question_start_col,
                question_end_col=question_end_col,
            )

            configs["partial_correlation"] = part_corr

        except Exception as e:
            print(f"Skipping partial correlation for {group_value}: {e}")

        for method_name, matrix in configs.items():
            plot_name = f"{method_name}_{group_value}"

            title = (
                f"{method_name.replace('_', ' ').title()} "
                f"Network — {group_value}"
            )

            save_correlation_matrix(
                matrix,
                output_dir / f"{plot_name}.pkl",
            )

            save_correlation_matrix(
                matrix,
                output_dir / f"{plot_name}.csv",
            )

            plot_correlation_matrix(
                matrix,
                title=(
                    f"{method_name.replace('_', ' ').title()} "
                    f"Matrix — {group_value}"
                ),
                output_path=plot_dir / f"{plot_name}_matrix.png",
                show=False,
            )

            plot_correlation_distribution(
                matrix,
                title=(
                    f"{method_name.replace('_', ' ').title()} "
                    f"Distribution — {group_value}"
                ),
                output_path=plot_dir / f"{plot_name}_distribution.png",
                show=False,
            )

            plot_correlation_network(
                matrix,
                threshold=0,
                title=title,
                output_path=plot_dir / f"{plot_name}_network.png",
                show=False,
            )

            group_results[method_name] = {
                "matrix": matrix,
            }

        results[group_value] = group_results

    return results