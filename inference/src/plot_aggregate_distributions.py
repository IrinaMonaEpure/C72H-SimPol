"""Plot aggregate 6x4 grid of correlation distributions for all countries.

Three methods overlaid per subplot:
  1. Correlation (Kendall) — from data
  2. Partial correlation (GraphicalLassoCV) — from data
  3. Peixoto (PseudoNormalBlockState) — from inferred network

Usage (from inference/src/):
    python plot_aggregate_distributions.py

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
        ../output/networks/<cntry>.pkl
Writes: ../output/figures/aggregate_distributions.{png,pdf}
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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

FONTSIZE = 13
ALPHA = 0.5
KDE_BW = 0.15
BINS = 30

COLORS = {
    "Correlation": "#4C72B0",
    "Partial Correlation": "#DD8452",
    "Peixoto": "#55A868",
}

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}


def off_diag_upper(matrix):
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    vals = matrix[mask]
    return vals[~np.isnan(vals)]


def peixoto_partial_corr(bn):
    W = bn.state.get_precision().todense()
    W = np.asarray(W)
    D = np.sqrt(np.diag(W))
    n = W.shape[0]
    pc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pc[i, j] = -W[i, j] / (D[i] * D[j])
    return pc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)

    pkl_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(NETWORK_DIR, "*.pkl"))
    }

    country_codes = sorted(
        c for c in df["cntry"].dropna().unique()
        if c in COUNTRY_NAMES and c in pkl_paths
    )

    xlim = (-0.5, 0.5)
    nrows, ncols = 6, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 20))
    axes = axes.flatten()

    ymax = 0
    country_data = []

    for code in country_codes:
        group = df[df["cntry"] == code]
        X = group[BELIEF_DIMS].apply(pd.to_numeric, errors="coerce")

        corr_vals = off_diag_upper(X.dropna().corr(method="kendall").to_numpy())

        X_imputed = SimpleImputer(strategy="median").fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X_imputed)
        model = GraphicalLassoCV()
        model.fit(X_scaled)
        precision = model.precision_
        pc = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
        np.fill_diagonal(pc, 0)
        pc_vals = off_diag_upper(pc)

        with open(pkl_paths[code], "rb") as f:
            bn = pickle.load(f)
        peixoto_vals = off_diag_upper(peixoto_partial_corr(bn))

        entry = {"code": code, "corr": corr_vals, "partial": pc_vals, "peixoto": peixoto_vals}
        country_data.append(entry)

        for vals in (corr_vals, pc_vals, peixoto_vals):
            counts, _ = np.histogram(vals, bins=BINS, range=xlim)
            ymax = max(ymax, counts.max())

        print(f"  {code} done")

    ymax_hist = int(min(ymax * 1.1, max(ymax + 5, 110)))

    x_kde = np.linspace(xlim[0], xlim[1], 300)
    bin_edges = np.linspace(xlim[0], xlim[1], BINS + 1)
    full_bar_width = bin_edges[1] - bin_edges[0]
    n_methods = 3
    group_width = full_bar_width * 0.8
    bar_width = group_width / n_methods
    methods = [("Correlation", "corr"), ("Partial Correlation", "partial"), ("Peixoto", "peixoto")]

    ymax_density = 0
    for entry in country_data:
        for _, key in methods:
            kde = gaussian_kde(entry[key], bw_method=KDE_BW)
            ymax_density = max(ymax_density, kde(x_kde).max())
    ymax_density *= 1.1

    for show_bars in (True, False):
        for ax in axes:
            ax.clear()
            ax.set_visible(True)

        cur_ymax = ymax_hist if show_bars else ymax_density
        ylabel = "Frequency" if show_bars else "Density"

        for idx, entry in enumerate(country_data):
            ax = axes[idx]
            code = entry["code"]
            name = COUNTRY_NAMES.get(code, code)

            for mi, (method, key) in enumerate(methods):
                vals = entry[key]
                color = COLORS[method]

                if show_bars:
                    counts, _ = np.histogram(vals, bins=bin_edges)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    offsets = bin_centers - group_width / 2 + mi * bar_width + bar_width / 2
                    ax.bar(
                        offsets, counts, width=bar_width, alpha=ALPHA,
                        color=color, edgecolor=color, linewidth=0.3,
                    )

                kde = gaussian_kde(vals, bw_method=KDE_BW)
                density = kde(x_kde)
                if show_bars:
                    scaled = density * len(vals) * full_bar_width
                    ax.plot(x_kde, scaled, color=color, linewidth=1.2)
                else:
                    ax.plot(x_kde, density, color=color, linewidth=1.5)

            ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")
            ax.set_xlim(*xlim)
            ax.set_ylim(0, cur_ymax)
            ax.set_title(f"{name} ({code})", fontsize=FONTSIZE, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=FONTSIZE)

        legend_ax = axes[len(country_data)]
        legend_ax.axis("off")
        if show_bars:
            handles = [
                mpatches.Patch(facecolor=COLORS[m], alpha=0.7, edgecolor=COLORS[m], label=m)
                for m in ["Correlation", "Partial Correlation", "Peixoto"]
            ]
        else:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=COLORS[m], linewidth=2.5, label=m)
                for m in ["Correlation", "Partial Correlation", "Peixoto"]
            ]
        legend_ax.legend(
            handles=handles, loc="center", fontsize=FONTSIZE * 1.2,
            frameon=False, handlelength=2, handleheight=1.5,
        )

        for idx in range(len(country_data) + 1, nrows * ncols):
            axes[idx].set_visible(False)

        fig.supxlabel("Weight", fontsize=FONTSIZE * 1.25)
        fig.supylabel(ylabel, fontsize=FONTSIZE * 1.25)
        fig.tight_layout(rect=[0.03, 0.02, 1, 0.98])

        suffix = "bars" if show_bars else "density"
        for fmt in ("png", "pdf"):
            out_path = os.path.join(OUT_DIR, f"aggregate_distributions_{suffix}.{fmt}")
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
