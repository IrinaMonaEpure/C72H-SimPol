"""Generate matrix + distribution figures for the two data-driven methods,
matching the style of plot_all_networks.py (which produces the Peixoto figures).

Methods:
  1. Correlation (Kendall)              -> <code>_correlation_{matrix,distribution}.png
  2. Partial correlation (GraphicalLassoCV) -> <code>_partial_{matrix,distribution}.png

These feed the webapp method selector alongside the existing Peixoto figures
(<code>_{matrix,distribution}.png from plot_all_networks.py).

Usage (from inference/src/):
    python plot_webapp_methods.py

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
Writes: ../output/figures/<code>_{correlation,partial}_{matrix,distribution}.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import GraphicalLassoCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_CSV = "../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv"
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

# Real-data belief names -> short labels (matches plot_three_networks.py)
LABEL_MAP = {
    "left_right_identification": "left_right", "gender_inequality": "gender_ineq",
    "anti_lgbt": "anti_lgbt", "euroscepticism": "eurosceptic",
    "anti_immigration": "anti_immig", "anti_egalitarianism": "anti_egal",
    "benefits_harm_economy": "harm_econ", "benefits_harm_society": "harm_soc",
    "welfare_chauvinism": "welf_chauv", "anti_economic_interventionism": "anti_interv",
    "anti_social_benefits_low_income": "anti_lowinc",
    "anti_social_benefits_parents": "anti_parents",
    "educational_spending": "anti_edu", "anti_basic_income": "anti_ubi",
    "anti_climate_change_taxes": "anti_clim_tax",
    "anti_climate_change_renewables": "anti_renew",
    "anti_climate_ban_appliances": "anti_clim_ban", "climate_skepticism": "clim_skeptic",
    "authoritarianism": "authoritarian", "anti_libertarianism": "anti_libert",
}

# Human-readable method names for figure titles
METHOD_TITLE = {
    "correlation": "Correlation (Kendall)",
    "partial": "Partial Correlation",
}


def correlation_matrix(X):
    """Kendall correlation matrix from the belief columns (drop NA rows).

    The diagonal (all 1.0) is left as-is: it is masked in the matrix heatmap
    and excluded (k=1) from the distribution, so it never reaches a plot.
    """
    return X.dropna().corr(method="kendall")


def partial_correlation_matrix(X):
    """GraphicalLassoCV partial correlation matrix (median-imputed, scaled)."""
    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)
    model = GraphicalLassoCV()
    model.fit(X_scaled)
    precision = model.precision_
    pc = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(pc, 0.0)
    return pd.DataFrame(pc, index=X.columns, columns=X.columns)


def plot_matrix(corr, title, output_path, cbar_label):
    # Mask upper triangle AND zero entries (non-edges) — same as plot_all_networks
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask |= (np.abs(corr.values) < 1e-10)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"label": cbar_label}, ax=ax,
    )
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(corr, title, output_path, xlabel, bins=40,
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
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
    df = pd.read_csv(DATA_CSV)
    codes = sorted(c for c in df["cntry"].dropna().unique() if c in COUNTRY_NAMES)
    print(f"Found {len(codes)} countries\n")

    # Compute all matrices first so distribution y-limits can be shared per method
    computed = {"correlation": {}, "partial": {}}
    for code in codes:
        group = df[df["cntry"] == code]
        X = group[BELIEF_DIMS].apply(pd.to_numeric, errors="coerce")
        X = X.rename(columns=lambda c: LABEL_MAP.get(c, c))

        computed["correlation"][code] = correlation_matrix(X)
        try:
            computed["partial"][code] = partial_correlation_matrix(X)
        except Exception as e:  # noqa: BLE001
            print(f"  partial correlation failed for {code}: {e}")
        print(f"{code} ({COUNTRY_NAMES[code]}): done")

    for method in ("correlation", "partial"):
        mats = computed[method]
        ylim = compute_shared_ylim(list(mats.values()))
        cbar = "Correlation" if method == "correlation" else "Partial correlation"
        xlabel = f"{METHOD_TITLE[method]} value"
        for code, corr in mats.items():
            name = COUNTRY_NAMES[code]
            plot_matrix(
                corr,
                title=f"{METHOD_TITLE[method]} Matrix — {name} ({code})",
                output_path=os.path.join(OUT_DIR, f"{code}_{method}_matrix.png"),
                cbar_label=cbar,
            )
            plot_distribution(
                corr,
                title=f"{METHOD_TITLE[method]} Distribution — {name} ({code})",
                output_path=os.path.join(OUT_DIR, f"{code}_{method}_distribution.png"),
                xlabel=xlabel,
                ylim=ylim,
            )

    print(f"\nDone. Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
