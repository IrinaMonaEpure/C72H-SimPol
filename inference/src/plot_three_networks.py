"""Plot network visualizations for a country using three methods side by side.

Methods:
  1. Correlation (Kendall) — from data
  2. Partial correlation (GraphicalLassoCV) — from data
  3. Peixoto (PseudoNormalBlockState) — from inferred network

Usage (from inference/src/):
    python plot_three_networks.py [COUNTRY_CODE]
    python plot_three_networks.py RU

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
        ../output/networks/<cntry>.pkl
Writes: ../output/figures/three_networks_<cntry>.{png,pdf}
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_CSV = "../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv"
NETWORK_DIR = "../output/networks"
OUT_DIR = "../output/figures"

# The 20 belief dimensions, selected by name (same as generate_all_networks.py)
BELIEF_DIMS = [
    "left_right_identification", "gender_inequality", "anti_lgbt",
    "euroscepticism", "anti_immigration", "anti_egalitarianism",
    "benefits_harm_economy", "benefits_harm_society", "welfare_chauvinism",
    "anti_economic_interventionism", "anti_social_benefits_low_income",
    "anti_social_benefits_parents", "educational_spending",
    "anti_basic_income", "anti_climate_change_taxes",
    "anti_climate_change_renewables", "anti_climate_ban_appliances",
    "climate_skepticism", "authoritarianism", "anti_libertarianism",
]

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}

LABEL_MAP = {
    "left_right": "left_right",
    "gender_ineq": "gender_ineq",
    "anti_lgbt": "anti_lgbt",
    "eurosceptic": "eurosceptic",
    "anti_immigration": "anti_immig",
    "anti_egalitarian": "anti_egal",
    "benefits_harm_economy": "harm_econ",
    "benefits_harm_society": "harm_soc",
    "welfare_chauvinism": "welf_chauv",
    "anti_interventionism": "anti_interv",
    "anti_benefits_lowinc": "anti_lowinc",
    "anti_benefits_parents": "anti_parents",
    "anti_edu_spending": "anti_edu",
    "anti_basic_income": "anti_ubi",
    "anti_climate_tax": "anti_clim_tax",
    "anti_climate_renew": "anti_renew",
    "anti_climate_ban": "anti_clim_ban",
    "climate_skeptic": "clim_skeptic",
    "authoritarian": "authoritarian",
    "anti_libertarian": "anti_libert",
}

PEIXOTO_LABEL_MAP = {
    "left_right_identification": "left_right",
    "gender_inequality": "gender_ineq",
    "anti_lgbt": "anti_lgbt",
    "euroscepticism": "eurosceptic",
    "anti_immigration": "anti_immig",
    "anti_egalitarianism": "anti_egal",
    "benefits_harm_economy": "harm_econ",
    "benefits_harm_society": "harm_soc",
    "welfare_chauvinism": "welf_chauv",
    "anti_economic_interventionism": "anti_interv",
    "anti_social_benefits_low_income": "anti_lowinc",
    "anti_social_benefits_parents": "anti_parents",
    "educational_spending": "anti_edu",
    "anti_basic_income": "anti_ubi",
    "anti_climate_change_taxes": "anti_clim_tax",
    "anti_climate_change_renewables": "anti_renew",
    "anti_climate_ban_appliances": "anti_clim_ban",
    "climate_skepticism": "clim_skeptic",
    "authoritarianism": "authoritarian",
    "anti_libertarianism": "anti_libert",
}


def matrix_to_graph(matrix, labels):
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n):
        G.add_node(labels[i])
    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i, j]
            if w != 0 and not np.isnan(w):
                G.add_edge(labels[i], labels[j], weight=w)
    return G


def draw_network(G, ax, title, layout_seed=42):
    ax.set_aspect("equal")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.set_title(title, fontsize=12, fontweight="bold")
        return

    pos = nx.spring_layout(
        G, seed=layout_seed, k=1.2, iterations=500, weight="weight"
    )

    coords = np.array([pos[node] for node in G.nodes()])
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = cmax - cmin
    span[span == 0] = 1
    coords = 2 * (coords - cmin) / span - 1

    x, y = coords[:, 0], coords[:, 1]
    node_to_i = {node: i for i, node in enumerate(G.nodes())}

    pad = 0.15
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)

    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        color = "#d62728" if weight < 0 else "#1f77b4"
        lw = np.clip(abs(weight) * 8, 0.4, 4.0)
        u_i, v_i = node_to_i[u], node_to_i[v]
        ax.plot(
            [x[u_i], x[v_i]], [y[u_i], y[v_i]],
            color=color, linewidth=lw, alpha=0.6,
            zorder=1, solid_capstyle="round",
        )

    ax.scatter(
        x, y, s=200, color="white", edgecolors="#333333",
        linewidths=1.5, zorder=2,
    )

    cx, cy = x.mean(), y.mean()
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))

    for node in G.nodes():
        i = node_to_i[node]
        dx, dy = x[i] - cx, y[i] - cy
        norm = max(np.hypot(dx, dy), 1e-6)
        offset = 0.03 * span
        lx = x[i] + dx / norm * offset
        ly = y[i] + dy / norm * offset
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(
            lx, ly, str(node), fontsize=7,
            ha=ha, va=va, zorder=3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(title, fontsize=12, fontweight="bold")


def main():
    country_code = sys.argv[1] if len(sys.argv) > 1 else "RU"
    country_name = COUNTRY_NAMES.get(country_code, country_code)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    group = df[df["cntry"] == country_code]
    X = group[BELIEF_DIMS].apply(pd.to_numeric, errors="coerce")
    col_names = X.columns.tolist()
    short_labels = [PEIXOTO_LABEL_MAP.get(c, c) for c in col_names]

    # --- Method 1: Kendall correlation ---
    corr = X.dropna().corr(method="kendall").to_numpy()
    G_corr = matrix_to_graph(corr, short_labels)

    # --- Method 2: Partial correlation (GraphicalLassoCV) ---
    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)
    model = GraphicalLassoCV()
    model.fit(X_scaled)
    precision = model.precision_
    pc = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(pc, 0)
    G_partial = matrix_to_graph(pc, short_labels)

    # --- Method 3: Peixoto ---
    pkl_path = os.path.join(NETWORK_DIR, f"{country_code}.pkl")
    with open(pkl_path, "rb") as f:
        bn = pickle.load(f)
    W = bn.state.get_precision().todense()
    W = np.asarray(W)
    D = np.sqrt(np.diag(W))
    n = W.shape[0]
    peixoto_pc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                peixoto_pc[i, j] = -W[i, j] / (D[i] * D[j])
    peixoto_labels = [PEIXOTO_LABEL_MAP.get(c, c) for c in bn.columns]
    G_peixoto = matrix_to_graph(peixoto_pc, peixoto_labels)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    plt.tight_layout()
    labs = ['A', 'B', 'C']
    for idx, ax in enumerate(axes):
        ax.text(-0.15, 1.05, labs[idx], transform=ax.transAxes, fontname='Arial',
                    fontsize=25, fontweight='bold', va='top', ha='left')

    draw_network(G_corr, axes[0], "Correlation (Kendall)")
    draw_network(G_partial, axes[1], "Partial Correlation")
    draw_network(G_peixoto, axes[2], "Peixoto")

    # fig.suptitle(
    #     f"{country_name} ({country_code}) — Belief Network Comparison\n"
    #     "Blue = positive  |  Red = negative",
    #     fontsize=14, fontweight="bold", y=1.02,
    # )
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"three_networks_{country_code}.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
