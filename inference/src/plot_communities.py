"""Plot hierarchical community structure for all inferred belief networks.

Runs nested SBM community detection on each inferred network and produces
radial hierarchy visualizations using graph-tool's draw_hierarchy.

Usage (from inference/src/):
    python plot_communities.py

Reads : ../output/networks/<cntry>.pkl
Writes: ../output/figures/<code>_communities.{pdf,png}
"""

import glob
import os
import pickle
import time
from multiprocessing import Pool, cpu_count

import numpy as np
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


def plot_country(args):
    code, pkl_path = args
    name = COUNTRY_NAMES.get(code, code)
    t0 = time.time()

    with open(pkl_path, "rb") as f:
        bn = pickle.load(f)

    g = bn.state.u.copy()

    state = gt.minimize_nested_blockmodel_dl(
        g, state_args=dict(recs=[bn.state.x], rec_types=["real-normal"])
    )

    levels = state.get_levels()
    b0 = levels[0].get_blocks()
    n_communities = len(set(int(b0[v]) for v in g.vertices()))

    vlabel = g.new_vp("string")
    for v in g.vertices():
        vlabel[v] = bn.columns[int(v)]

    W = bn.state.get_precision().todense()
    W = np.asarray(W)
    D = np.sqrt(np.diag(W))

    ecolor = g.new_ep("vector<double>")
    ewidth = g.new_ep("double")
    for e in g.edges():
        i, j = int(e.source()), int(e.target())
        pc = -W[i, j] / (D[i] * D[j])
        if pc > 0:
            ecolor[e] = [0.12, 0.47, 0.71, 0.5]
        else:
            ecolor[e] = [0.84, 0.15, 0.16, 0.5]
        ewidth[e] = max(0.3, abs(pc) * 6)

    for fmt in ("png", "pdf"):
        out_path = os.path.join(OUT_DIR, f"{code}_communities.{fmt}")
        output_size = (1200, 1200) if fmt == "png" else None

        gt.draw_hierarchy(
            state,
            layout="radial",
            vprops={"text": vlabel, "font_size": 10},
            eprops={"color": ecolor, "pen_width": ewidth},
            output=out_path,
            output_size=output_size,
        )

    elapsed = time.time() - t0
    print(f"  {code} ({name}): {n_communities} communities, "
          f"{len(levels)} levels, {elapsed:.1f}s")
    return {
        "country": code,
        "n_communities": n_communities,
        "n_levels": len(levels),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(NETWORK_DIR, "*.pkl")))
    if not paths:
        print("No .pkl files found. Run generate_all_networks.py first.")
        return

    print(f"Found {len(paths)} pickled networks\n")

    tasks = []
    for path in paths:
        code = os.path.splitext(os.path.basename(path))[0]
        tasks.append((code, path))

    n_workers = min(len(tasks), cpu_count())
    print(f"Plotting with {n_workers} parallel workers\n")

    t_total = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(plot_country, tasks)
    t_total = time.time() - t_total

    print(f"\nDone in {t_total:.0f}s. Figures saved to {OUT_DIR}/")
    for r in results:
        print(f"  {r['country']}: {r['n_communities']} communities, "
              f"{r['n_levels']} levels")


if __name__ == "__main__":
    main()
