import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
from graph_tool.inference import PseudoNormalBlockState


class BeliefNetwork:
    """Infer a weighted belief network from survey data using Peixoto's MDL method.

    Uses the multivariate Gaussian generative model with the nested SBM prior
    (PseudoNormalBlockState from graph-tool).
    """

    def __init__(self, columns):
        """
        Parameters
        ----------
        columns : list of str
            Column names in the DataFrame that correspond to belief dimensions.
            Each column becomes a node in the inferred network.
        """
        self.columns = list(columns)
        self.state = None
        self.graph = None
        self.weights = None
        self.dl_history = []

    def fit(self, df, max_iter=1000, patience=50, niter_per_sweep=10,
            verbose=True):
        """Run MDL inference on the data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all columns specified at construction.
            Missing values are mean-imputed per column.
        max_iter : int
            Maximum number of outer MCMC iterations.
        patience : int
            Stop if description length doesn't improve for this many iterations.
        niter_per_sweep : int
            Number of inner MCMC steps per outer iteration.
        verbose : bool
            Print progress every 100 iterations.

        Returns
        -------
        self
        """
        X = self._prepare_data(df)

        self.state = PseudoNormalBlockState(
            X, nested=True, fix_mean=True, positive=True
        )

        self.dl_history = [self.state.entropy()]
        best_dl = self.dl_history[0]
        no_improve = 0

        for i in range(max_iter):
            self.state.mcmc_sweep(niter=niter_per_sweep)
            dl = self.state.entropy()
            self.dl_history.append(dl)

            if dl < best_dl:
                best_dl = dl
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (i + 1) % 100 == 0:
                n_edges = self.state.u.num_edges()
                print(f"iter {i+1:4d} | DL = {dl:.1f} | edges = {n_edges}")

            if no_improve >= patience:
                if verbose:
                    print(f"Converged at iteration {i+1}")
                break

        self.graph = self.state.u
        self.weights = self.state.x

        if verbose:
            print(f"\nFinal description length : {self.state.entropy():.1f}")
            print(f"Inferred edges           : {self.graph.num_edges()}")

        return self

    def get_edge_list(self):
        """Return a DataFrame of inferred edges with their weights.

        Returns
        -------
        pd.DataFrame
            Columns: source, target, weight
        """
        self._check_fitted()
        rows = []
        for e in self.graph.edges():
            rows.append({
                "source": self.columns[int(e.source())],
                "target": self.columns[int(e.target())],
                "weight": self.weights[e],
            })
        return pd.DataFrame(rows)

    def plot_convergence(self, ax=None):
        """Plot the description length over MCMC iterations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if not provided.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.dl_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Description length")
        ax.set_title("MDL convergence")
        return ax

    def plot_network(self, title=None, ax=None, figsize=(14, 14),
                     layout_C=10, layout_gamma=0.5):
        """Draw the inferred network with matplotlib.

        Nodes are laid out with graph-tool's SFDP algorithm. Edges are coloured
        blue (positive weight) or red (negative weight), with thickness
        proportional to absolute weight.

        Parameters
        ----------
        title : str, optional
            Plot title. A default is used if not provided.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if not provided.
        figsize : tuple
            Figure size if ax is not provided.
        layout_C : float
            Repulsion strength for SFDP layout.
        layout_gamma : float
            Cooling factor for SFDP layout.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        g = self.graph
        w = self.weights

        pos = gt.sfdp_layout(g, C=layout_C, gamma=layout_gamma,
                             p=2, max_iter=5000)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        coords = np.array([pos[v] for v in g.vertices()])
        x, y = coords[:, 0], coords[:, 1]

        for e in g.edges():
            u_i, v_i = int(e.source()), int(e.target())
            weight = w[e]
            color = "#d62728" if weight < 0 else "#1f77b4"
            lw = np.clip(abs(weight) * 4.0, 0.4, 4.0)
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
            ax.text(lx, ly, self.columns[i], fontsize=9,
                    ha=ha, va=va, zorder=3, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.7))

        if title is None:
            title = ("Inferred belief network (MDL, multivariate Gaussian)\n"
                     "Blue = positive  |  Red = negative")
        ax.set_title(title, fontsize=11)
        return ax

    def get_partial_correlations(self):
        """Return partial correlations derived from the inferred precision matrix.

        Converts raw precision matrix entries W_ij to partial correlations
        via: rho_ij = -W_ij / sqrt(W_ii * W_jj).

        Returns
        -------
        gt.EdgePropertyMap
            Edge property map with partial correlation values.
        """
        self._check_fitted()
        W = self.state.get_precision().todense()
        W = np.asarray(W)
        D = np.sqrt(np.diag(W))

        pc = self.graph.new_ep("double")
        for e in self.graph.edges():
            i, j = int(e.source()), int(e.target())
            pc[e] = -W[i, j] / (D[i] * D[j])
        return pc

    def save(self, path):
        """Save the inferred network to a file (GraphML or GT format).

        Node property 'belief' stores the belief dimension name.
        Edge property 'weight' stores partial correlations.

        Parameters
        ----------
        path : str
            Output file path. Format is inferred from extension
            (.graphml, .xml, or .gt).
        """
        self._check_fitted()
        g = self.graph.copy()

        vlabel = g.new_vp("string")
        for v in g.vertices():
            vlabel[v] = self.columns[int(v)]
        g.vp["belief"] = vlabel

        pc = self.get_partial_correlations()
        eweight = g.new_ep("double")
        for e_new, e_old in zip(g.edges(), self.graph.edges()):
            eweight[e_new] = pc[e_old]
        g.ep["weight"] = eweight

        g.save(path)

    def _prepare_data(self, df):
        """Mean-impute and transpose to (N_nodes, M_samples)."""
        data = df[self.columns].copy()
        data = data.fillna(data.mean())
        return data.values.T

    def _check_fitted(self):
        if self.state is None:
            raise RuntimeError("Call .fit() before using this method.")
