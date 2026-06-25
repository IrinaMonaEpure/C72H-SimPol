import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Core measure (same as in plot_phase_diagram.py) ──────────────────────────

def polarization_var_pairwise(belief_matrix: np.ndarray) -> float:
    """Variance of pairwise Euclidean distances across agents in belief space."""
    sq_norms = (belief_matrix ** 2).sum(axis=1)
    dot      = belief_matrix @ belief_matrix.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot
    sq_dists = np.clip(sq_dists, 0, None)
    dists    = np.sqrt(sq_dists)
    i_up, j_up = np.triu_indices(belief_matrix.shape[0], k=1)
    return float(dists[i_up, j_up].var())


def polarization_time_series(belief_history: np.ndarray) -> np.ndarray:
    """
    Compute polarization at every snapshot.

    Parameters
    ----------
    belief_history : (n_snapshots, n_agents, n_beliefs)

    Returns
    -------
    (n_snapshots,) array
    """
    return np.array([
        polarization_var_pairwise(belief_history[t])
        for t in range(belief_history.shape[0])
    ])


# ── Main plot function ────────────────────────────────────────────────────────

def plot_polarization_trajectories(
    countries,
    network_type   = "correlation",
    results_root   = "results",
    save_path      = None,
    figsize_per_cell = (5, 3.5),
):
    """
    2×2 grid of polarization-over-time plots, one line per country.

    Layout
    ------
    Rows    → β_internal  (high on top, low on bottom)
    Columns → β_external  (low on left,  high on right)

    Parameters
    ----------
    countries        : list of country codes matching result folder names,
                       e.g. ["GB", "IT", "BE"]
    network_type     : "correlation" | "partial_correlation" | "mdl"
    results_root     : root folder where experiment subfolders live
    save_path        : if given, saves the figure here (e.g. "polarization_grid.png")
    figsize_per_cell : (width, height) of each subplot cell in inches
    """
    # Infer type_tag from NETWORK_CONFIGS if available, else derive it
    try:
        type_tag = NETWORK_CONFIGS[network_type]["type_tag"]
    except (NameError, KeyError):
        _tags = {"correlation": "corr", "partial_correlation": "pcorr", "mdl": "mdl"}
        type_tag = _tags.get(network_type, network_type)

    results_root = Path(results_root)

    # ── Discover beta grid from the first country's manifest ─────────────────
    first_exp_dir = results_root / f"{countries[0].lower()}_{type_tag}_all_beta_grid"
    manifest0     = json.loads((first_exp_dir / "manifest.json").read_text())

    beta_internals = sorted(set(r["beta_internal"] for r in manifest0["runs"]))
    beta_externals = sorted(set(r["beta_external"] for r in manifest0["runs"]))
    n_bi = len(beta_internals)
    n_be = len(beta_externals)

    # ── Build a lookup: (bi, be) → run metadata, per country ─────────────────
    # country_runs[country][(bi, be)] = (steps, polarization_series)
    country_series = {}

    for country in countries:
        exp_dir  = results_root / f"{country.lower()}_{type_tag}_all_beta_grid"
        manifest = json.loads((exp_dir / "manifest.json").read_text())

        series = {}
        for run in manifest["runs"]:
            bi = run["beta_internal"]
            be = run["beta_external"]

            data = np.load(exp_dir / run["result_file"], allow_pickle=False)
            steps         = data["steps"]                  # (n_snapshots,)
            belief_history = data["belief_history"]        # (n_snapshots, n_agents, n_beliefs)

            pol = polarization_time_series(belief_history)  # (n_snapshots,)
            series[(bi, be)] = (steps, pol)

        country_series[country] = series

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_bi, n_be,
        figsize=(figsize_per_cell[0] * n_be, figsize_per_cell[1] * n_bi),
        sharex=True,
        sharey=True,
        squeeze=False,       # always returns 2-D array
    )

    # Row 0 = highest β_internal (reversed for visual: high on top)
    bi_order = list(reversed(beta_internals))   # [high, …, low]
    be_order = beta_externals                   # [low, …, high]

    for row, bi in enumerate(bi_order):
        for col, be in enumerate(be_order):
            ax = axes[row, col]

            for k, country in enumerate(countries):
                steps, pol = country_series[country][(bi, be)]
                ax.plot(steps, pol, color=f"C{k}", linewidth=1.4, label=country)

            ax.set_title(
                f"β_int = {bi:.2g}   β_soc = {be:.2g}",
                fontsize=10, pad=4,
            )
            ax.grid(True, alpha=0.25)

            # Axis labels on outer edges only
            if col == 0:
                ax.set_ylabel("Polarization\n(var. pairwise dist.)", fontsize=8)
            if row == n_bi - 1:
                ax.set_xlabel("Simulation step", fontsize=9)

    # ── Shared legend above the grid ─────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(countries),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"Belief polarisation over time  —  {network_type}",
        fontsize=12,
        y=1.06,
    )
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


# ── Usage ─────────────────────────────────────────────────────────────────────
