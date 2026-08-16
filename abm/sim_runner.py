"""Sparse opinion-exchange simulation runner built on the Cython kernel
(direct_social_belief_sim_cy_sparse), plus beta-grid experiment orchestration
that snapshots simulation state and saves results to disk.
"""

from __future__ import annotations

import itertools
import json
import random as _random
from datetime import datetime
from pathlib import Path

import igraph as ig
import numpy as np
from tqdm.auto import tqdm

import direct_social_belief_sim_cy_sparse as sim_cy
from boccaletti2007 import generate_boccaletti_graph

DEFAULT_BELIEF_VALUES = np.linspace(-1.0, 1.0, 7)


_IGRAPH_RNG_ROUTED = False


def _seed_igraph_rng(seed: int | None) -> None:
    """Make igraph's built-in stochastic generators (Erdos-Renyi, Barabasi-
    Albert, Watts-Strogatz, configuration model) reproducible.

    igraph has its own internal RNG, separate from `random`/`numpy.random`.
    Routing it through the stdlib `random` module (igraph's documented
    mechanism for this) lets `random.seed(seed)` control it.
    """
    global _IGRAPH_RNG_ROUTED
    if not _IGRAPH_RNG_ROUTED:
        ig.set_random_number_generator(_random)
        _IGRAPH_RNG_ROUTED = True
    if seed is not None:
        _random.seed(seed)


def _generate_boccaletti(n_nodes, seed=None, *, m=3, n0=None, initial_graph=None, **_ignored):
    """Nonhierarchical scale-free growth model (Boccaletti et al. 2007)."""
    return generate_boccaletti_graph(n_nodes=n_nodes, m=m, n0=n0, initial_graph=initial_graph, seed=seed)


def _generate_complete(n_nodes, seed=None, **_ignored):
    """Complete graph — every pair of agents is connected."""
    return ig.Graph.Full(n_nodes, directed=False, loops=False)


def _generate_erdos_renyi(n_nodes, seed=None, *, p=None, m=None, **_ignored):
    """Erdos-Renyi random graph. Give either p (edge probability) or m (edge count)."""
    if p is None and m is None:
        raise ValueError("erdos_renyi network requires either 'p' (edge probability) or 'm' (edge count)")
    _seed_igraph_rng(seed)
    if p is not None:
        return ig.Graph.Erdos_Renyi(n=n_nodes, p=p, directed=False, loops=False)
    return ig.Graph.Erdos_Renyi(n=n_nodes, m=m, directed=False, loops=False)


def _generate_configuration_model(n_nodes, seed=None, *, degree_sequence=None,
                                   method="configuration_simple", **_ignored):
    """Configuration model from an explicit degree sequence (length n_nodes).

    method="configuration_simple" (default) rejects self-loops/multi-edges so
    the result is a simple graph, which is what the agent-influence model
    assumes; pass method="configuration" for the classical multigraph version.
    """
    if degree_sequence is None:
        raise ValueError("configuration_model network requires a 'degree_sequence' of length n_nodes")
    degree_sequence = list(degree_sequence)
    if len(degree_sequence) != n_nodes:
        raise ValueError(f"degree_sequence must have length n_nodes={n_nodes}, got {len(degree_sequence)}")
    _seed_igraph_rng(seed)
    return ig.Graph.Degree_Sequence(degree_sequence, method=method)


def _generate_barabasi_albert(n_nodes, seed=None, *, m=3, power=1.0, **_ignored):
    """Barabasi-Albert preferential-attachment growth model."""
    _seed_igraph_rng(seed)
    return ig.Graph.Barabasi(n=n_nodes, m=m, power=power, directed=False)


def _generate_watts_strogatz(n_nodes, seed=None, *, nei=2, p=0.1, **_ignored):
    """Watts-Strogatz small-world model: ring lattice (radius `nei`) rewired
    with probability `p`."""
    _seed_igraph_rng(seed)
    return ig.Graph.Watts_Strogatz(dim=1, size=n_nodes, nei=nei, p=p)


NETWORK_GENERATORS = {
    "boccaletti": _generate_boccaletti,
    "complete": _generate_complete,
    "erdos_renyi": _generate_erdos_renyi,
    "configuration_model": _generate_configuration_model,
    "barabasi_albert": _generate_barabasi_albert,
    "watts_strogatz": _generate_watts_strogatz,
}


def generate_agent_network(network_type: str, n_nodes: int, seed: int | None = None, **kwargs) -> ig.Graph:
    """Generate an undirected agent social network.

    Parameters
    ----------
    network_type : one of NETWORK_GENERATORS.keys() —
        "boccaletti"           : m (edges per new node, default 3), n0, initial_graph
        "complete"              : (no extra parameters)
        "erdos_renyi"           : p (edge probability) or m (edge count)
        "configuration_model"   : degree_sequence (length n_nodes), method
        "barabasi_albert"       : m (edges per new node, default 3), power (default 1.0)
        "watts_strogatz"        : nei (neighborhood radius, default 2), p (rewiring prob, default 0.1)
    n_nodes : number of agents
    seed    : random seed for reproducible generation
    """
    try:
        generator = NETWORK_GENERATORS[network_type]
    except KeyError:
        raise ValueError(
            f"Unknown network_type {network_type!r}. Choose from {sorted(NETWORK_GENERATORS)}"
        ) from None
    return generator(n_nodes=n_nodes, seed=seed, **kwargs)


def agent_network_tag(n_agents: int, network_type: str) -> str:
    """Short tag identifying the agent population size and social-network
    topology, for use in result/plot file and folder names."""
    return f"{n_agents}agents_{network_type}"


def build_agent_graph(n_agents: int, m: int, n_beliefs: int, seed: int, *,
                       network_type: str = "boccaletti", **network_kwargs) -> ig.Graph:
    """Generate an agent social network and attach uniformly-sampled initial beliefs.

    network_type selects the topology (see generate_agent_network for the
    full list and their extra parameters, passed via network_kwargs). `m` is
    the edges-per-new-node parameter for "boccaletti" and "barabasi_albert";
    it is ignored by the other network types.
    """
    if network_type in ("boccaletti", "barabasi_albert"):
        network_kwargs.setdefault("m", m)

    g = generate_agent_network(network_type, n_nodes=n_agents, seed=seed, **network_kwargs)

    rng = np.random.default_rng(seed)
    initial_beliefs = rng.choice(DEFAULT_BELIEF_VALUES, size=(g.vcount(), n_beliefs))
    g.vs["beliefs"] = initial_beliefs.tolist()
    return g


def belief_graph_to_matrix(belief_graph, weight_attr="weight", default_weight=1.0, zero_diagonal=True):
    if isinstance(belief_graph, np.ndarray):
        W = np.asarray(belief_graph, dtype=np.float64).copy()
    elif hasattr(belief_graph, "toarray"):
        W = np.asarray(belief_graph.toarray(), dtype=np.float64)
    elif isinstance(belief_graph, ig.Graph):
        n = belief_graph.vcount()
        W = np.zeros((n, n), dtype=np.float64)
        has_weight = weight_attr in belief_graph.es.attributes()

        for edge in belief_graph.es:
            i, j = edge.tuple
            w = float(edge[weight_attr]) if has_weight and edge[weight_attr] is not None else default_weight
            W[i, j] = w
            if not belief_graph.is_directed():
                W[j, i] = w
    else:
        raise TypeError("belief_graph must be a NumPy matrix, scipy sparse matrix, or igraph.Graph")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("belief graph/weights must be a square matrix")

    if zero_diagonal:
        np.fill_diagonal(W, 0.0)

    return W


def agent_graph_to_csr_tuple(agent_graph, weight_attr="weight", default_weight=1.0):
    if not isinstance(agent_graph, ig.Graph):
        raise TypeError("agent_graph must be an igraph.Graph")

    rows, cols, data = [], [], []
    has_weight = weight_attr in agent_graph.es.attributes()

    for edge in agent_graph.es:
        i, j = edge.tuple
        w = float(edge[weight_attr]) if has_weight and edge[weight_attr] is not None else default_weight

        rows.append(i)
        cols.append(j)
        data.append(w)

        if not agent_graph.is_directed():
            rows.append(j)
            cols.append(i)
            data.append(w)

    n = agent_graph.vcount()
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    data = np.asarray(data, dtype=np.float64)

    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)

    return indptr, cols, data, (n, n)


def run_sparse_simulation_with_snapshots(
    agent_graph,
    belief_graph,
    *,
    beta_internal,
    beta_external,
    number_of_steps,
    history_every,
    focal_beliefs=None,
    belief_attr="beliefs",
    allowed_values=None,
    normalize_neighbor_influence=False,
    seed=42,
    show_progress=True,
):
    belief_weights = belief_graph_to_matrix(belief_graph)

    initial_beliefs = np.asarray(agent_graph.vs[belief_attr], dtype=np.float64)
    if initial_beliefs.ndim != 2:
        raise ValueError(f'agent_graph.vs["{belief_attr}"] must contain belief vectors')

    n_agents, n_beliefs = initial_beliefs.shape
    if belief_weights.shape != (n_beliefs, n_beliefs):
        raise ValueError("belief_graph size must match belief vector length")

    agent_csr = agent_graph_to_csr_tuple(agent_graph)
    current_beliefs = initial_beliefs.copy()

    history = [
        {
            "step": 0,
            "beliefs": current_beliefs.copy(),
            "mean": float(current_beliefs.mean()),
        }
    ]

    seed_rng = np.random.default_rng(seed)
    starts = range(0, number_of_steps, history_every)

    if show_progress:
        starts = tqdm(starts, desc="Simulation snapshots")

    for start in starts:
        chunk_steps = min(history_every, number_of_steps - start)
        chunk_seed = None if seed is None else int(seed_rng.integers(1, 2**63 - 1))

        config = sim_cy.ExchangeConfig(
            beta_internal=beta_internal,
            beta_social=beta_external,
            focal_beliefs=focal_beliefs,
            n_steps=chunk_steps,
            seed=chunk_seed,
        )

        graph_chunk, _ = sim_cy.run_exchange_fast_sparse(
            belief_weights,
            agent_csr,
            allowed_values=allowed_values,
            initial_beliefs=current_beliefs,
            config=config,
            normalize_neighbor_influence=normalize_neighbor_influence,
            return_history=False,
            show_progress=False,
        )

        current_beliefs = np.asarray(graph_chunk.vs["beliefs"], dtype=np.float64)

        history.append(
            {
                "step": start + chunk_steps,
                "beliefs": current_beliefs.copy(),
                "mean": float(current_beliefs.mean()),
            }
        )

    final_graph = agent_graph.copy()
    final_graph.vs[belief_attr] = [row.tolist() for row in current_beliefs]

    return final_graph, history


def run_beta_grid_experiment(
    agent_graph,
    belief_graph,
    param_grid,
    *,
    focal_beliefs,
    number_of_steps,
    history_every,
    results_root="results",
    experiment_name=None,
    seed=42,
    allowed_values=None,
    normalize_neighbor_influence=False,
    show_progress=True,
    overwrite=False,
):
    if "beta_internal" not in param_grid or "beta_external" not in param_grid:
        raise ValueError('param_grid must contain "beta_internal" and "beta_external"')

    focal_beliefs_saved = None if focal_beliefs is None else [int(x) for x in focal_beliefs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name is None:
        experiment_name = f"beta_grid_{timestamp}"

    results_root = Path(results_root)
    results_dir = results_root / experiment_name

    if results_dir.exists() and not overwrite:
        results_dir = results_root / f"{experiment_name}_{timestamp}"

    results_dir.mkdir(parents=True, exist_ok=overwrite)

    beta_internal_values = list(param_grid["beta_internal"])
    beta_external_values = list(param_grid["beta_external"])

    combinations = list(
        itertools.product(
            beta_internal_values,
            beta_external_values,
        )
    )

    manifest = {
        "experiment_name": experiment_name,
        "created_at": timestamp,
        "results_dir": str(results_dir),
        "number_of_steps": int(number_of_steps),
        "history_every": int(history_every),
        "focal_beliefs": focal_beliefs_saved,
        "param_grid": {
            "beta_internal": beta_internal_values,
            "beta_external": beta_external_values,
        },
        "runs": [],
    }

    iterator = enumerate(combinations)
    if show_progress:
        iterator = tqdm(iterator, total=len(combinations), desc="Beta grid")

    for run_idx, (beta_internal, beta_external) in iterator:
        run_seed = None if seed is None else int(seed + run_idx)

        final_graph, history = run_sparse_simulation_with_snapshots(
            agent_graph,
            belief_graph,
            beta_internal=float(beta_internal),
            beta_external=float(beta_external),
            number_of_steps=int(number_of_steps),
            history_every=int(history_every),
            focal_beliefs=focal_beliefs,
            allowed_values=allowed_values,
            normalize_neighbor_influence=normalize_neighbor_influence,
            seed=run_seed,
            show_progress=False,
        )

        steps = np.asarray([item["step"] for item in history], dtype=np.int64)
        means = np.asarray([item["mean"] for item in history], dtype=np.float64)
        belief_history = np.stack([item["beliefs"] for item in history]).astype(np.float64)
        final_beliefs = np.asarray(final_graph.vs["beliefs"], dtype=np.float64)

        run_meta = {
            "run_idx": int(run_idx),
            "beta_internal": float(beta_internal),
            "beta_external": float(beta_external),
            "focal_beliefs": focal_beliefs_saved,
            "number_of_steps": int(number_of_steps),
            "history_every": int(history_every),
            "seed": run_seed,
            "n_snapshots": int(len(history)),
            "result_file": None,
        }

        file_name = (
            f"run_{run_idx:04d}"
            f"_bi_{float(beta_internal):.6g}"
            f"_be_{float(beta_external):.6g}"
            ".npz"
        )
        file_path = results_dir / file_name
        run_meta["result_file"] = file_name

        np.savez_compressed(
            file_path,
            steps=steps,
            means=means,
            belief_history=belief_history,
            final_beliefs=final_beliefs,
            focal_beliefs=np.asarray(
                [] if focal_beliefs_saved is None else focal_beliefs_saved,
                dtype=np.int64,
            ),
            beta_internal=np.asarray(float(beta_internal)),
            beta_external=np.asarray(float(beta_external)),
            seed=np.asarray(-1 if run_seed is None else run_seed, dtype=np.int64),
            metadata=json.dumps(run_meta, ensure_ascii=False),
        )

        manifest["runs"].append(run_meta)

    manifest_path = results_dir / "manifest.json"
    manifest["manifest_file"] = str(manifest_path)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results_dir, manifest
