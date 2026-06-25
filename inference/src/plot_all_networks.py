"""Plot belief networks for all countries from saved GraphML files.

Usage (from inference/src/):
    python plot_all_networks.py

Reads : ../output/networks/<cntry>.graphml
Writes: ../output/figures/<cntry>.pdf  and  <cntry>.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt

NETWORK_DIR = "../output/networks"
OUT_DIR = "../output/figures"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "UK": "United Kingdom",
}


def plot_network(g, beliefs, weights, title, ax):
    pos = gt.sfdp_layout(g, C=10, gamma=0.5, p=2, max_iter=5000)

    ax.set_aspect("equal")
    ax.axis("off")

    coords = np.array([pos[v] for v in g.vertices()])
    x, y = coords[:, 0], coords[:, 1]

    for e in g.edges():
        u_i, v_i = int(e.source()), int(e.target())
        w = weights[e]
        color = "#d62728" if w < 0 else "#1f77b4"
        lw = np.clip(abs(w) * 4.0, 0.4, 4.0)
        ax.plot([x[u_i], x[v_i]], [y[u_i], y[v_i]],
                color=color, linewidth=lw, alpha=0.6,
                zorder=1, solid_capstyle="round")

    ax.scatter(x, y, s=300, color="white", edgecolors="#333333",
               linewidths=1.5, zorder=2)

    cx, cy = x.mean(), y.mean()
    span = coords.max() - coords.min()
    for v in g.vertices():
        i = int(v)
        dx, dy = x[i] - cx, y[i] - cy
        norm = max(np.hypot(dx, dy), 1e-6)
        offset = 0.02 * span
        lx = x[i] + dx / norm * offset
        ly = y[i] + dy / norm * offset
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(lx, ly, beliefs[v], fontsize=9,
                ha=ha, va=va, zorder=3, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="none", alpha=0.7))

    ax.set_title(title, fontsize=11)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(NETWORK_DIR, "*.graphml")))
    print(f"Found {len(paths)} network files\n")

    for path in paths:
        code = os.path.splitext(os.path.basename(path))[0]
        name = COUNTRY_NAMES.get(code, code)

        g = gt.load_graph(path)
        beliefs = g.vp["belief"]
        weights = g.ep["weight"]

        n_edges = g.num_edges()
        print(f"{code} ({name}): {n_edges} edges")

        title = (f"Inferred belief network — {name} ({code})\n"
                 f"Blue = positive  |  Red = negative  |  {n_edges} edges")

        fig, ax = plt.subplots(figsize=(14, 14))
        plot_network(g, beliefs, weights, title, ax)
        plt.tight_layout()

        fig.savefig(os.path.join(OUT_DIR, f"{code}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUT_DIR, f"{code}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"\nDone. Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
