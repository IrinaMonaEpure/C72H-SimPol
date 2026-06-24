from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class GrowthStep:
    """One recorded step of the Boccaletti-Hwang-Latora growth rule."""

    new_node: int
    anchor: int
    candidates: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class GraphSummary:
    """Basic structural diagnostics for a generated graph."""

    n_nodes: int
    n_edges: int
    mean_degree: float
    clustering: float
    assortativity: float


def generate_boccaletti_graph(
    n_nodes: int,
    m: int,
    *,
    n0: int | None = None,
    initial_graph: ig.Graph | None = None,
    seed: int | None = None,
    show_progress: bool = False,
    return_history: bool = False,
) -> ig.Graph | tuple[ig.Graph, list[GrowthStep]]:
    """Generate the nonhierarchical scale-free network from Boccaletti et al. (2007).

    The model starts from a connected graph whose nodes have degree at least
    ``m``. By default this is the complete graph on ``n0 = m + 1`` nodes. At
    each growth step a new node is added and connected to ``m`` existing nodes:

    1. Choose an existing anchor node ``j`` uniformly at random.
    2. Build the local candidate set ``S_j = {j} + neighbors(j)``.
    3. Choose ``m`` distinct nodes uniformly from ``S_j`` and connect the new
       node to them.

    No degree-proportional attachment probability is written into the rule.
    High-degree nodes still receive more links because they appear in more
    candidate sets ``S_j``.

    Args:
        n_nodes: Final number of graph vertices.
        m: Number of edges attached to every newly added vertex.
        n0: Size of the default complete seed graph. If omitted, ``m + 1`` is
            used, the smallest complete graph satisfying the model constraint.
        initial_graph: Optional custom undirected seed graph. Every seed node
            must have degree at least ``m`` so each local candidate set has
            enough targets.
        seed: Random seed for reproducible growth.
        show_progress: Whether to show a tqdm progress bar.
        return_history: Whether to return recorded local choices for each
            added node. This can be memory-heavy for large graphs.

    Returns:
        The generated undirected igraph.Graph, or ``(graph, history)`` when
        ``return_history`` is True.
    """

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if m <= 0:
        raise ValueError("m must be positive.")

    rng = np.random.default_rng(seed)
    graph = _make_seed_graph(n_nodes, m, n0=n0, initial_graph=initial_graph)
    history: list[GrowthStep] = []

    steps = range(graph.vcount(), n_nodes)
    if show_progress:
        steps = tqdm(steps, desc="Boccaletti growth", leave=False)

    for new_node in steps:
        existing_count = graph.vcount()
        anchor = int(rng.integers(existing_count))
        candidates = (anchor, *graph.neighbors(anchor))

        if len(candidates) < m:
            raise ValueError(
                f"Anchor {anchor} has only {len(candidates)} local candidates, "
                f"but m={m} targets are required."
            )

        targets_array = rng.choice(candidates, size=m, replace=False)
        targets = tuple(int(target) for target in targets_array)

        graph.add_vertices(1)
        graph.add_edges((int(new_node), target) for target in targets)

        if return_history:
            history.append(
                GrowthStep(
                    new_node=int(new_node),
                    anchor=anchor,
                    candidates=tuple(int(node) for node in candidates),
                    targets=targets,
                )
            )

    if return_history:
        return graph, history
    return graph


def degree_distribution(graph: ig.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return degree values and their probabilities."""

    degrees = np.asarray(graph.degree(), dtype=int)
    counts = np.bincount(degrees)
    nonzero = counts > 0
    return np.flatnonzero(nonzero), counts[nonzero] / graph.vcount()


def clustering_by_degree(graph: ig.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean local clustering coefficient for each degree class."""

    degrees = np.asarray(graph.degree(), dtype=int)
    local_clustering = np.asarray(graph.transitivity_local_undirected(mode="zero"), dtype=float)

    degree_values = np.unique(degrees)
    clustering = np.empty(degree_values.size, dtype=float)
    for idx, degree in enumerate(degree_values):
        clustering[idx] = float(local_clustering[degrees == degree].mean())
    return degree_values, clustering


def summary(graph: ig.Graph) -> GraphSummary:
    """Compute quick diagnostics used in the Boccaletti et al. paper."""

    degrees = np.asarray(graph.degree(), dtype=float)
    return GraphSummary(
        n_nodes=graph.vcount(),
        n_edges=graph.ecount(),
        mean_degree=float(degrees.mean()) if degrees.size else 0.0,
        clustering=float(graph.transitivity_avglocal_undirected(mode="zero")),
        assortativity=float(graph.assortativity_degree(directed=False)),
    )


def adjacency_matrix(graph: ig.Graph, *, dtype: type = float) -> np.ndarray:
    """Convert an igraph graph to a dense NumPy adjacency matrix."""

    return np.asarray(graph.get_adjacency().data, dtype=dtype)


def _make_seed_graph(
    n_nodes: int,
    m: int,
    *,
    n0: int | None,
    initial_graph: ig.Graph | None,
) -> ig.Graph:
    if initial_graph is not None:
        graph = initial_graph.copy()
        if graph.is_directed():
            raise ValueError("initial_graph must be undirected.")
        if graph.vcount() > n_nodes:
            raise ValueError("initial_graph cannot have more vertices than n_nodes.")
        if graph.vcount() == 0:
            raise ValueError("initial_graph must contain at least one vertex.")
        if not graph.is_connected():
            raise ValueError("initial_graph must be connected.")
        if min(graph.degree()) < m:
            raise ValueError("Every initial_graph vertex must have degree at least m.")
        return graph

    if n0 is None:
        n0 = m + 1
    if n0 < m + 1:
        raise ValueError("n0 must be at least m + 1 for the complete seed graph.")
    if n0 > n_nodes:
        raise ValueError("n0 cannot be larger than n_nodes.")
    return ig.Graph.Full(n0, directed=False, loops=False)


if __name__ == "__main__":
    graph = generate_boccaletti_graph(n_nodes=10_000, m=5, n0=6, seed=42, show_progress=True)
    print(summary(graph))

    degrees, probabilities = degree_distribution(graph)
    print("First degree classes:")
    for degree, probability in zip(degrees[:10], probabilities[:10]):
        print(f"k={degree:2d}  P(k)={probability:.5f}")
