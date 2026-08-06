"""Box plot comparing near-zero value fractions across three methods.

Methods:
  1. Correlation (Kendall) — from data
  2. Partial correlation (GraphicalLassoCV) — from data
  3. Peixoto (PseudoNormalBlockState) — from inferred network

Usage (from inference/src/):
    python plot_near_zero_boxplot.py

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
        ../output/networks/<cntry>.pkl
Writes: ../output/figures/near_zero_boxplot.{png,pdf}
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV

DATA_CSV = "../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv"
NETWORK_DIR = "../output/networks"
OUT_DIR = "../output/figures"
ZERO_THRESHOLD = 0.05

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

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}


def off_diag_values(matrix):
    A = np.asarray(matrix, dtype=float)
    mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    vals = A[mask]
    return vals[~np.isnan(vals)]


def near_zero_fraction(values, threshold=ZERO_THRESHOLD):
    return np.mean(np.abs(values) <= threshold)


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
    countries = sorted(df["cntry"].dropna().unique())

    pkl_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(NETWORK_DIR, "*.pkl"))
    }

    rows = []

    for code in countries:
        if code not in COUNTRY_NAMES or code not in pkl_paths:
            continue

        group = df[df["cntry"] == code]
        X = group[BELIEF_DIMS]
        X = X.apply(pd.to_numeric, errors="coerce")

        corr = X.dropna().corr(method="kendall").to_numpy()
        corr_vals = off_diag_values(corr)
        rows.append({
            "country": code,
            "method": "Correlation",
            "near_zero_fraction": near_zero_fraction(corr_vals),
        })

        try:
            X_imputed = SimpleImputer(strategy="median").fit_transform(X)
            X_scaled = StandardScaler().fit_transform(X_imputed)
            model = GraphicalLassoCV()
            model.fit(X_scaled)
            precision = model.precision_
            pc = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
            np.fill_diagonal(pc, 0)
            pc_vals = off_diag_values(pc)
            rows.append({
                "country": code,
                "method": "Partial Correlation",
                "near_zero_fraction": near_zero_fraction(pc_vals),
            })
        except Exception as e:
            print(f"  Skipping partial correlation for {code}: {e}")

        with open(pkl_paths[code], "rb") as f:
            bn = pickle.load(f)
        peixoto_pc = peixoto_partial_corr(bn)
        peixoto_vals = off_diag_values(peixoto_pc)
        rows.append({
            "country": code,
            "method": "Peixoto",
            "near_zero_fraction": near_zero_fraction(peixoto_vals),
        })

        print(f"  {code} done")

    result = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ["Correlation", "Partial Correlation", "Peixoto"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    box_data = [
        result[result["method"] == m]["near_zero_fraction"].values
        for m in methods
    ]

    bp = ax.boxplot(
        box_data,
        labels=methods,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (m, color) in enumerate(zip(methods, colors)):
        vals = result[result["method"] == m]["near_zero_fraction"].values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full_like(vals, i + 1) + jitter, vals,
            color=color, edgecolors="white", s=40, zorder=3, linewidths=0.5,
        )

    ax.set_ylabel(
        f"Fraction of edges with |weight| ≤ {ZERO_THRESHOLD}",
        fontsize=FONTSIZE,
    )
    ax.set_xlabel("Method", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"near_zero_boxplot.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
