import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Reuse from plot_phase_diagram / plot_polarization_trajectories ────────────

def polarization_var_pairwise(belief_matrix: np.ndarray) -> float:
    """Variance of pairwise Euclidean distances across agents in belief space."""
    sq_norms = (belief_matrix ** 2).sum(axis=1)
    dot      = belief_matrix @ belief_matrix.T
    sq_dists = np.clip(sq_norms[:, None] + sq_norms[None, :] - 2 * dot, 0, None)
    dists    = np.sqrt(sq_dists)
    i_up, j_up = np.triu_indices(belief_matrix.shape[0], k=1)
    return float(dists[i_up, j_up].var())


# ── Type-tag helper (mirrors NETWORK_CONFIGS) ─────────────────────────────────

def _type_tag(network_type: str) -> str:
    return {"correlation": "corr", "partial_correlation": "pcorr", "mdl": "mdl"}.get(
        network_type, network_type
    )


# ── Main function ─────────────────────────────────────────────────────────────

def plot_correlation_vs_polarization(
    network_type,
    x_column,
    beta_internal = 2.0,
    beta_external = 0.5,
    results_root  = "results",
    csv_dir       = ".",
    save_path     = None,
    ax            = None,
):
    """
    Scatter plot: signed correlation sum (x) vs final-step polarization (y),
    one point per country.

    Parameters
    ----------
    network_type  : "correlation" | "partial_correlation" | "mdl"
    x_column      : column name in the CSV to use as x-axis values,
                    e.g. "signed_correlation_sum"
    beta_internal : β_internal value to read results for (default 2.0)
    beta_external : β_external value to read results for (default 0.5)
    results_root  : root folder containing experiment subfolders
    csv_dir       : folder containing correlation_sums_***.csv files
    save_path     : if given, saves the figure here
    ax            : existing Axes to draw on; if None a new figure is created

    Returns
    -------
    fig, ax
    """
    tag          = _type_tag(network_type)
    results_root = Path(results_root)
    csv_path     = Path(csv_dir) / f"correlation_sums_{tag}.csv"

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)   # first column = country codes
    df.index = df.index.str.strip().str.upper()

    if x_column not in df.columns:
        raise ValueError(
            f"Column '{x_column}' not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── For each country, load the matching run and compute polarization ───────
    x_vals, y_vals, labels = [], [], []
    missing = []

    for country in df.index:
        exp_dir = results_root / f"{country.lower()}_{tag}_all_beta_grid"
        manifest_path = exp_dir / "manifest.json"

        if not manifest_path.exists():
            missing.append(country)
            continue

        manifest = json.loads(manifest_path.read_text())

        # Find the run matching the requested beta combination
        run = next(
            (
                r for r in manifest["runs"]
                if abs(r["beta_internal"] - beta_internal) < 1e-9
                and abs(r["beta_external"] - beta_external) < 1e-9
            ),
            None,
        )
        if run is None:
            missing.append(country)
            continue

        data          = np.load(exp_dir / run["result_file"], allow_pickle=False)
        final_beliefs = data["belief_history"][-1]   # (n_agents, n_beliefs)
        pol           = polarization_var_pairwise(final_beliefs)

        x_vals.append(float(df.loc[country, x_column]))
        y_vals.append(pol)
        labels.append(country)

    if missing:
        print(f"⚠ No results found for: {missing} — skipped.")

    if not x_vals:
        raise RuntimeError("No data to plot — check results_root and network_type.")

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(x_vals, y_vals, s=60, zorder=3)

    # Annotate each point with its country code
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(x_column.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("Polarization  (var. pairwise dist.)", fontsize=10)
    ax.set_title(
        f"{network_type}  —  β_int={beta_internal:.2g},  β_soc={beta_external:.2g}",
        fontsize=11,
    )
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig, ax


# ── Usage ─────────────────────────────────────────────────────────────────────


