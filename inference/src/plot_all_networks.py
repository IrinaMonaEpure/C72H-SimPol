"""Plot belief networks for all countries from saved GraphML files.

Generates three plot types per country:
  1. Correlation matrix heatmap    — <code>_matrix.{pdf,png}
  2. Correlation distribution      — <code>_distribution.{pdf,png}
  3. Correlation network graph     — <code>_network.{pdf,png}

Usage (from inference/src/):
    python plot_all_networks.py

Reads : ../output/networks/<cntry>.graphml
Writes: ../output/figures/<code>_{matrix,distribution,network}.{pdf,png}
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import graph_tool.all as gt

NETWORK_DIR = "../output/networks"
OUT_DIR = "../output/figures"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}


def graphml_to_corr_matrix(path):
    g = gt.load_graph(path)
    beliefs = g.vp["belief"]
    weights = g.ep["weight"]

    N = g.num_vertices()
    labels = [beliefs[v] for v in g.vertices()]

    matrix = np.zeros((N, N))
    for e in g.edges():
        i, j = int(e.source()), int(e.target())
        matrix[i, j] = weights[e]
        matrix[j, i] = weights[e]

    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_correlation_matrix(corr, title, output_path):
    # Mask upper triangle AND zero entries (non-edges)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask |= (np.abs(corr.values) < 1e-10)
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        corr, mask=mask, cmap="RdBu", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Partial correlation"}, ax=ax,
    )
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_distribution(corr, title, output_path, bins=40,
                                  xlim=(-1, 1), ylim=None):
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    values = corr.to_numpy()[upper]
    values = values[~np.isnan(values)]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, bins=bins, binrange=xlim, kde=True, ax=ax)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Partial correlation value")
    ax.set_ylabel("Count")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_network(corr, title, output_path, threshold=0,
                             layout_k=1.2, layout_iterations=500, seed=42):
    adjacency = corr.copy()
    adjacency = adjacency.mask(np.eye(len(adjacency), dtype=bool), 0)
    adjacency[np.abs(adjacency) < threshold] = 0

    G = nx.from_pandas_adjacency(adjacency)
    G.remove_edges_from([(u, v) for u, v, w in G.edges(data="weight") if w == 0])
    G.remove_nodes_from(list(nx.isolates(G)))

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.set_title("No edges above threshold", fontsize=11)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(G, seed=seed, k=layout_k,
                           iterations=layout_iterations, weight="weight")

    coords = np.array([pos[node] for node in G.nodes()])
    x, y = coords[:, 0], coords[:, 1]
    node_to_i = {node: i for i, node in enumerate(G.nodes())}

    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        color = "#d62728" if weight < 0 else "#1f77b4"
        lw = np.clip(abs(weight) * 8, 0.4, 4.0)
        u_i, v_i = node_to_i[u], node_to_i[v]
        ax.plot([x[u_i], x[v_i]], [y[u_i], y[v_i]],
                color=color, linewidth=lw, alpha=0.6,
                zorder=1, solid_capstyle="round")

    ax.scatter(x, y, s=300, color="white", edgecolors="#333333",
               linewidths=1.5, zorder=2)

    cx, cy = x.mean(), y.mean()
    span = max(coords[:, 0].max() - coords[:, 0].min(),
               coords[:, 1].max() - coords[:, 1].min())

    for node in G.nodes():
        i = node_to_i[node]
        dx, dy = x[i] - cx, y[i] - cy
        norm = max(np.hypot(dx, dy), 1e-6)
        offset = 0.03 * span
        lx = x[i] + dx / norm * offset
        ly = y[i] + dy / norm * offset
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(lx, ly, str(node), fontsize=9, ha=ha, va=va, zorder=3,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))

    ax.set_title(title, fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_shared_ylim(matrices, bins=40, xlim=(-1, 1)):
    ymax = 0
    for matrix in matrices:
        upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        values = matrix.to_numpy()[upper]
        values = values[~np.isnan(values)]
        counts, _ = np.histogram(values, bins=bins, range=xlim)
        if len(counts) > 0:
            ymax = max(ymax, counts.max())
    return (0, ymax * 1.1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(NETWORK_DIR, "*.graphml")))
    print(f"Found {len(paths)} network files\n")

    matrices = {}
    for path in paths:
        code = os.path.splitext(os.path.basename(path))[0]
        matrices[code] = graphml_to_corr_matrix(path)

    distribution_ylim = compute_shared_ylim(list(matrices.values()))

    for code, corr in matrices.items():
        name = COUNTRY_NAMES.get(code, code)
        n_edges = int((np.abs(corr.values) > 1e-10).sum() - corr.shape[0]) // 2
        print(f"{code} ({name}): {n_edges} edges")

        for fmt in ("png", "pdf"):
            plot_correlation_matrix(
                corr,
                title=f"Partial Correlation Matrix — {name} ({code})",
                output_path=os.path.join(OUT_DIR, f"{code}_matrix.{fmt}"),
            )

            plot_correlation_distribution(
                corr,
                title=f"Partial Correlation Distribution — {name} ({code})",
                output_path=os.path.join(OUT_DIR, f"{code}_distribution.{fmt}"),
                ylim=distribution_ylim,
            )

            plot_correlation_network(
                corr,
                title=(f"Partial Correlation Network — {name} ({code})\n"
                       f"Blue = positive  |  Red = negative  |  {n_edges} edges"),
                output_path=os.path.join(OUT_DIR, f"{code}_network.{fmt}"),
            )

    print(f"\nDone. Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
