import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_belief_dynamics(belief_history, steps, chosen_beliefs, run_label=None, belief_names=None):
    """
    Layout
    ------
    Row 0 : Legend (spanning both columns)
    Row 1 : Mean of all chosen beliefs over time (spanning both columns)
    Row 2+: [Variance over time] | [Agent-belief distribution: initial vs final]
            one row per belief

    Parameters
    ----------
    belief_history : np.ndarray, shape (n_snapshots, n_agents, n_beliefs)
    steps          : np.ndarray, shape (n_snapshots,)
    chosen_beliefs : list[int]   — belief indices to plot
    run_label      : str | None  — optional title suffix (e.g. "β_i=1.0, β_e=1.5")
    belief_names   : list[str] | None — display names matching chosen_beliefs;
                     if None, labels default to "Belief {index}"
    """
    chosen_beliefs = list(chosen_beliefs)
    n_chosen = len(chosen_beliefs)

    if belief_names is not None:
        if len(belief_names) != n_chosen:
            raise ValueError(
                f"belief_names has {len(belief_names)} entries but chosen_beliefs has {n_chosen}."
            )
        labels = list(belief_names)
    else:
        labels = [f"Belief {b}" for b in chosen_beliefs]

    # ── Compute statistics over the agent axis ───────────────────────────────
    subset    = belief_history[:, :, chosen_beliefs]  # (T, N, K)
    means     = subset.mean(axis=1)                   # (T, K)
    variances = subset.var(axis=1)                    # (T, K)

    # ── Build GridSpec ───────────────────────────────────────────────────────
    # 2 header rows (legend + mean) + n_chosen data rows, 2 columns
    n_grid_rows  = 2 + n_chosen
    height_ratios = [0.35, 1.8] + [1.6] * n_chosen

    fig = plt.figure(figsize=(14, 1.5 + 2.8 + 2.2 * n_chosen))
    gs  = gridspec.GridSpec(
        n_grid_rows, 2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.30,
    )

    # ── Row 0: Legend ────────────────────────────────────────────────────────
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    handles = [
        plt.Line2D([0], [0], color=f"C{k}", linewidth=2, label=lbl)
        for k, lbl in enumerate(labels)
    ]
    ax_legend.legend(
        handles=handles,
        loc="center",
        ncol=min(n_chosen, 6),
        fontsize=8,
        frameon=False,
    )

    # ── Row 1: Mean (spanning both columns) ──────────────────────────────────
    ax_mean = fig.add_subplot(gs[1, :])
    for k, lbl in enumerate(labels):
        ax_mean.plot(steps, means[:, k], color=f"C{k}", linewidth=1.2)
    ax_mean.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_mean.set_ylabel("Mean belief (over agents)", fontsize=9)
    ax_mean.grid(True, alpha=0.25)
    title = "Belief dynamics over time"
    if run_label:
        title += f"  ({run_label})"
    ax_mean.set_title(title, fontsize=12)

    # ── Rows 2+: Variance (left) | Distribution initial vs final (right) ────
    ax_var_first = None   # used to sharex all variance panels

    for k, (b, lbl) in enumerate(zip(chosen_beliefs, labels)):
        color = f"C{k}"
        row   = 2 + k

        # -- Variance panel --------------------------------------------------
        share_kw = dict(sharex=ax_var_first) if ax_var_first is not None else {}
        ax_var = fig.add_subplot(gs[row, 0], **share_kw)
        if ax_var_first is None:
            ax_var_first = ax_var

        ax_var.plot(steps, variances[:, k], color=color, linewidth=1.2)
        ax_var.set_ylabel(f"Variance\n({lbl})", fontsize=8)
        ax_var.grid(True, alpha=0.25)

        # Hide x tick labels on all but the last variance panel
        if k < n_chosen - 1:
            plt.setp(ax_var.get_xticklabels(), visible=False)
        else:
            ax_var.set_xlabel("Simulation step", fontsize=9)

        # -- Distribution panel (initial vs final) ---------------------------
        ax_dist = fig.add_subplot(gs[row, 1])

        initial_vals = belief_history[0,  :, b]
        final_vals   = belief_history[-1, :, b]

        for vals, alpha_bar, time_label in [
            (initial_vals, 0.35, "initial"),
            (final_vals,   0.70, "final"),
        ]:
            weights = np.ones(len(vals)) / len(vals)
            ax_dist.hist(vals, bins=40, weights=weights, color=color,
                         alpha=alpha_bar, label=time_label)

        ax_dist.legend(fontsize=7, frameon=False,loc="upper center")
        ax_dist.set_xlabel(lbl, fontsize=8)
        ax_dist.set_ylabel("Frequency", fontsize=8)
        ax_dist.grid(True, alpha=0.25)
        ax_dist.tick_params(labelsize=7)

    fig.align_ylabels()
    return fig


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    results_dir = Path("results/ex_all_beta_grid")

    manifest = json.loads((results_dir / "manifest.json").read_text())
    run = next(
        r for r in manifest["runs"]
        if r["beta_internal"] == 0.5 and r["beta_external"] == 2
    )

    data = np.load(results_dir / run["result_file"], allow_pickle=False)
    steps          = data["steps"]
    belief_history = data["belief_history"]   # (n_snapshots, n_agents, n_beliefs)

    chosen_beliefs = list(range(20))

    fig = plot_belief_dynamics(
        belief_history,
        steps,
        chosen_beliefs,
        run_label=f"β_i={run['beta_internal']}, β_e={run['beta_external']}",
        belief_names=belief_names,   # defined elsewhere in your script
    )
    fig.savefig("belief_dynamics.png", dpi=150, bbox_inches="tight")
    plt.show()
