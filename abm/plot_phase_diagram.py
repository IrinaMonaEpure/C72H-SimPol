import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── Polarization measure ─────────────────────────────────────────────────────

def polarization_var_pairwise(belief_matrix: np.ndarray) -> float:
    """
    Variance of pairwise Euclidean distances in the full belief space.

    A polarized population (two tight clusters far apart) produces a bimodal
    pairwise-distance distribution — many small within-group distances and many
    large between-group distances — maximising this variance.
    Consensus drives it to 0; uniform spread without clustering keeps it low.

    Parameters
    ----------
    belief_matrix : (n_agents, n_beliefs)

    Returns
    -------
    float — polarization score (higher = more polarized)
    """
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj  (avoids an O(N^2 D) loop)
    sq_norms  = (belief_matrix ** 2).sum(axis=1)           # (N,)
    dot       = belief_matrix @ belief_matrix.T             # (N, N)
    sq_dists  = sq_norms[:, None] + sq_norms[None, :] - 2 * dot
    sq_dists  = np.clip(sq_dists, 0, None)                 # numerical safety
    dists     = np.sqrt(sq_dists)

    # Use upper triangle only (each pair counted once)
    i_upper, j_upper = np.triu_indices(belief_matrix.shape[0], k=1)
    return float(dists[i_upper, j_upper].var())


# ── Load results and build grid ──────────────────────────────────────────────

def build_polarization_grid(results_dir: Path, use_snapshot: int = -1):
    """
    Iterate over all runs in the manifest and compute the polarization score.

    Parameters
    ----------
    results_dir  : directory containing manifest.json and .npz result files
    use_snapshot : which time snapshot to evaluate (-1 = final, 0 = initial, etc.)

    Returns
    -------
    grid            : (n_bi, n_be) array of polarization scores
    beta_internals  : sorted list of β_internal values (row axis)
    beta_externals  : sorted list of β_external values (column axis)
    """
    manifest = json.loads((results_dir / "manifest.json").read_text())
    runs = manifest["runs"]

    beta_internals = sorted(set(r["beta_internal"] for r in runs))
    beta_externals = sorted(set(r["beta_external"] for r in runs))
    bi_idx = {v: i for i, v in enumerate(beta_internals)}
    be_idx = {v: i for i, v in enumerate(beta_externals)}

    grid = np.full((len(beta_internals), len(beta_externals)), np.nan)

    for run in runs:
        data = np.load(results_dir / run["result_file"], allow_pickle=False)
        # belief_history: (n_snapshots, n_agents, n_beliefs)
        beliefs = data["belief_history"][use_snapshot]   # (n_agents, n_beliefs)
        score = polarization_var_pairwise(beliefs)
        grid[bi_idx[run["beta_internal"]], be_idx[run["beta_external"]]] = score
        print(f"  β_i={run['beta_internal']:.2f}  β_e={run['beta_external']:.2f}  "
              f"polarization={score:.4f}")

    return grid, beta_internals, beta_externals


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_phase_diagram(grid, beta_internals, beta_externals, title=None,
                       vmin=None, vmax=None):
    """
    Plot the polarization grid as a heatmap with annotated cell values.

    Parameters
    ----------
    grid           : (n_bi, n_be) polarization scores; rows = β_internal, cols = β_external
    beta_internals : list of β_internal values (row axis, low → high = bottom → top)
    beta_externals : list of β_external values (column axis)
    title          : optional figure title
    vmin, vmax     : colourbar limits; if None, derived from the data
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Fall back to data range when limits are not supplied
    _vmin = np.nanmin(grid) if vmin is None else vmin
    _vmax = np.nanmax(grid) if vmax is None else vmax

    im = ax.imshow(
        grid,
        origin="lower",       # β_internal increases upward
        aspect="auto",
        cmap="plasma",
        vmin=_vmin,
        vmax=_vmax,
    )

    # Axis ticks
    ax.set_xticks(range(len(beta_externals)))
    ax.set_xticklabels(["low","high"])
    ax.set_yticks(range(len(beta_internals)))
    ax.set_yticklabels(["low","high"])
    ax.set_xlabel("β_soc  (social influence)", fontsize=11)
    ax.set_ylabel("β_int  (internal coherence)", fontsize=11)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Polarization\n(var. of pairwise distances)", fontsize=9)

    # Annotate each cell with its value (brightness relative to fixed scale)
    for i in range(len(beta_internals)):
        for j in range(len(beta_externals)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            brightness = (val - _vmin) / (_vmax - _vmin + 1e-12)
            txt_color = "white" if brightness < 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    ax.set_title(
        title or "Phase diagram — belief polarisation\n"
                 r"(Var of pairwise $\ell_2$ distances in belief space)",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = Path("results/ex_all_beta_grid")

    print("Computing polarization scores …")
    grid, beta_internals, beta_externals = build_polarization_grid(
        results_dir,
        use_snapshot=-1,   # -1 = final state; change to e.g. 50 for a mid-run snapshot
    )

    fig = plot_phase_diagram(grid, beta_internals, beta_externals)
    fig.savefig("phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.show()
