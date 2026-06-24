# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import igraph as ig
import numpy as np
from tqdm.auto import tqdm

cimport cython
cimport numpy as cnp
from libc.math cimport exp, fabs, isfinite
from libc.stdint cimport uint64_t


DEFAULT_BELIEF_VALUES = np.linspace(-1.0, 1.0, 7)


@dataclass(frozen=True)
class ExchangeConfig:
    """Parameters controlling one opinion-exchange simulation run.

    beta_internal scales the pressure to align beliefs inside one agent,
    beta_social scales pressure from neighboring agents on focal topics,
    focal_beliefs defines which belief components can carry social
    discussion, n_steps is the number of exchange events, show_progress
    toggles tqdm.auto, and seed makes runs reproducible.

    Attributes:
        beta_internal: Strength of pressure from internal belief consistency.
        beta_social: Strength of pressure from neighbors' beliefs.
        focal_beliefs: Belief indices where social discussion is active. If
            None, all belief dimensions are treated as focal.
        discussion_size: Deprecated compatibility field. The model now always
            picks exactly one topic per step and requires this to be 1.
        n_steps: Number of opinion-exchange events to simulate.
        show_progress: Whether to show a tqdm.auto progress bar.
        seed: Random seed for reproducible sampling.
    """

    beta_internal: float = 1.0
    beta_social: float = 1.0
    focal_beliefs: Sequence[int] | None = None
    discussion_size: int = 1
    n_steps: int = 1_000
    show_progress: bool = False
    seed: int | None = None


def normalize_allowed_values(
    n_beliefs: int,
    allowed_values: Mapping[int, Iterable[float]] | None = None,
) -> dict[int, np.ndarray]:
    """Build a complete value grid for every belief dimension.

    The caller may provide custom allowed states for only some beliefs;
    missing beliefs receive the default seven-point grid from -1 to 1.
    The result is normalized to NumPy arrays for fast sampling.

    Args:
        n_beliefs: Total number of belief dimensions per agent.
        allowed_values: Optional mapping from belief index to allowed states.
            Missing belief indices use DEFAULT_BELIEF_VALUES.

    Returns:
        Dictionary mapping every belief index to a 1D NumPy array of allowed
        values.
    """

    if allowed_values is None:
        return {i: DEFAULT_BELIEF_VALUES.copy() for i in range(n_beliefs)}

    normalized = {}
    for belief_idx in range(n_beliefs):
        values = allowed_values.get(belief_idx, DEFAULT_BELIEF_VALUES)
        values_array = np.asarray(list(values), dtype=float)
        if values_array.ndim != 1 or values_array.size == 0:
            raise ValueError(f"Allowed values for belief {belief_idx} must be a non-empty 1D sequence.")
        normalized[belief_idx] = values_array
    return normalized


def normalize_focal_beliefs(
    n_beliefs: int,
    focal_beliefs: Sequence[int] | None = None,
) -> np.ndarray:
    """Build a boolean mask of belief dimensions with active social discussion."""

    if focal_beliefs is None:
        return np.ones(n_beliefs, dtype=bool)

    mask = np.zeros(n_beliefs, dtype=bool)
    for belief_idx in focal_beliefs:
        idx = int(belief_idx)
        if idx < 0 or idx >= n_beliefs:
            raise ValueError(f"focal_beliefs contains out-of-range belief index {idx}.")
        mask[idx] = True
    return mask


def validate_single_topic_config(config: ExchangeConfig) -> None:
    """Reject legacy multi-topic discussion settings."""

    if getattr(config, "discussion_size", 1) != 1:
        raise ValueError("discussion_size is fixed to 1: each step now chooses exactly one topic.")


def build_agent_graph(
    agent_adjacency: np.ndarray | Sequence[Sequence[float]],
    initial_beliefs: np.ndarray | Sequence[Sequence[float]],
    *,
    directed: bool | None = None,
) -> ig.Graph:
    """Create an igraph agent network from adjacency and belief matrices.

    agent_adjacency defines who can influence whom. initial_beliefs is a
    matrix with shape (n_agents, n_beliefs). Each igraph vertex stores its
    agent_id and current belief vector in the vertex attribute "beliefs".
    Edge weights are copied from nonzero adjacency entries.

    Args:
        agent_adjacency: Square matrix of social connections between agents.
            Nonzero values become edge weights.
        initial_beliefs: Matrix of agents' initial belief vectors with shape
            (n_agents, n_beliefs).
        directed: Whether to create a directed graph. If None, direction is
            inferred from symmetry of agent_adjacency.

    Returns:
        igraph.Graph with vertex attributes "agent_id" and "beliefs", and edge
        attribute "weight" when edges exist.
    """

    adjacency = np.asarray(agent_adjacency, dtype=float)
    beliefs = np.asarray(initial_beliefs, dtype=float)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("agent_adjacency must be a square matrix.")
    if beliefs.ndim != 2 or beliefs.shape[0] != adjacency.shape[0]:
        raise ValueError("initial_beliefs must have one row per agent.")

    if directed is None:
        directed = not np.allclose(adjacency, adjacency.T)

    edges = []
    weights = []
    n_agents = adjacency.shape[0]
    for i in range(n_agents):
        js = range(n_agents) if directed else range(i + 1, n_agents)
        for j in js:
            if i == j:
                continue
            if adjacency[i, j] != 0:
                edges.append((i, j))
                weights.append(float(adjacency[i, j]))

    graph = ig.Graph(n=n_agents, edges=edges, directed=directed)
    graph.vs["agent_id"] = list(range(n_agents))
    graph.vs["beliefs"] = [row.tolist() for row in beliefs]
    if weights:
        graph.es["weight"] = weights
    return graph


def graph_belief_matrix(graph: ig.Graph) -> np.ndarray:
    """Read vertex belief vectors from an igraph graph as a 2D array.

    Args:
        graph: igraph.Graph whose vertices contain the "beliefs" attribute.

    Returns:
        Belief matrix with shape (n_agents, n_beliefs).
    """

    return np.asarray(graph.vs["beliefs"], dtype=float)


def set_graph_belief_matrix(graph: ig.Graph, beliefs: np.ndarray) -> None:
    """Write a 2D belief matrix back into the igraph vertex attributes.

    Args:
        graph: igraph.Graph to update.
        beliefs: New belief matrix with one row per graph vertex.
    """

    graph.vs["beliefs"] = [row.tolist() for row in np.asarray(beliefs, dtype=float)]


def internal_energy(
    agent_beliefs: np.ndarray,
    belief_weights: np.ndarray,
    discussion_beliefs: Sequence[int] | None = None,
) -> float:
    """Legacy partial internal dissonance over selected discussion beliefs.

    This helper preserves the earlier Q-restricted energy calculation for
    diagnostics and comparisons. The simulation runner now uses
    full_internal_energy, so discussion_size selects degrees of freedom
    without changing the effective internal-energy scale. No focal belief is
    assumed. For every selected belief i, its current value is compared with
    all beliefs j through belief_weights[i, j]. Lower energy means stronger
    alignment under positive weights.

    Args:
        agent_beliefs: Current belief vector of one agent.
        belief_weights: Square matrix of weights between belief dimensions.
        discussion_beliefs: Belief indices included in the current discussion.
            If None, all belief dimensions are included.

    Returns:
        Internal energy H_internal for the selected discussion beliefs.
    """
    beliefs = np.asarray(agent_beliefs, dtype=float)
    weights = np.asarray(belief_weights, dtype=float)
    if weights.shape != (beliefs.size, beliefs.size):
        raise ValueError("belief_weights must be square with size equal to the number of beliefs.")

    if discussion_beliefs is None:
        idx = np.arange(beliefs.size)
    else:
        idx = np.asarray(discussion_beliefs, dtype=int)

    return float(-np.sum(weights[np.ix_(idx, np.arange(beliefs.size))] * beliefs[idx, None] * beliefs[None, :]))


def full_internal_energy(
    agent_beliefs: np.ndarray,
    belief_weights: np.ndarray,
) -> float:
    """Full internal dissonance over all belief dimensions.

    This treats all belief coordinates as part of the internal belief system,
    independent of the current discussion set Q:
    H_internal = -sum_ab B_ab * x_a * x_b.
    """

    beliefs = np.asarray(agent_beliefs, dtype=float)
    weights = np.asarray(belief_weights, dtype=float)
    if weights.shape != (beliefs.size, beliefs.size):
        raise ValueError("belief_weights must be square with size equal to the number of beliefs.")
    return float(-np.sum(weights * np.outer(beliefs, beliefs)))


def social_energy(
    agent_idx: int,
    agent_beliefs: np.ndarray,
    all_beliefs: np.ndarray,
    agent_adjacency: np.ndarray,
    discussion_beliefs: Sequence[int],
    *,
    normalize_neighbor_influence: bool = False,
) -> float:
    """Social impact for selected beliefs aggregated over all neighbors.

    For every discussed belief, neighbors' current values are summed using
    the agent-adjacency weights. The selected agent pays lower energy when
    its discussed beliefs align with that weighted neighbor signal.

    Args:
        agent_idx: Index of the selected agent.
        agent_beliefs: Current belief vector of the selected agent.
        all_beliefs: Belief matrix for all agents.
        agent_adjacency: Square social adjacency matrix between agents.
        discussion_beliefs: Belief indices included in the current discussion.
        normalize_neighbor_influence: If True, divide neighbor weights by the
            sum of absolute neighbor weights so high-degree agents do not
            automatically receive stronger social pressure.

    Returns:
        Social energy H_social for the selected agent and discussion beliefs.
    """

    adjacency = np.asarray(agent_adjacency, dtype=float)
    selected = np.asarray(discussion_beliefs, dtype=int)
    neighbor_weights = adjacency[agent_idx].copy()
    neighbor_weights[agent_idx] = 0.0

    if normalize_neighbor_influence:
        denom = np.sum(np.abs(neighbor_weights))
        if denom > 0:
            neighbor_weights = neighbor_weights / denom

    neighbor_signal = neighbor_weights @ all_beliefs[:, selected]
    return float(-np.sum(agent_beliefs[selected] * neighbor_signal))


def pressure(
    agent_idx: int,
    agent_beliefs: np.ndarray,
    all_beliefs: np.ndarray,
    belief_weights: np.ndarray,
    agent_adjacency: np.ndarray,
    discussion_beliefs: Sequence[int],
    *,
    beta_internal: float,
    beta_social: float,
    include_social: bool = True,
    normalize_neighbor_influence: bool = False,
) -> float:
    """Combine internal and social energy into one update pressure.

    This is the direct-social analogue of felt dissonance:
    D = beta_internal * H_internal + beta_social * H_social.
    Lower D makes a candidate belief state more likely.

    Args:
        agent_idx: Index of the selected agent.
        agent_beliefs: Candidate or current belief vector for the selected
            agent.
        all_beliefs: Belief matrix for all agents.
        belief_weights: Square matrix of weights between belief dimensions.
        agent_adjacency: Square social adjacency matrix between agents.
        discussion_beliefs: Belief indices included in the current discussion.
        beta_internal: Strength of internal consistency pressure.
        beta_social: Strength of neighbor social pressure.
        normalize_neighbor_influence: Whether to normalize neighbor weights
            before aggregating social influence.

    Returns:
        Total pressure D for the supplied agent belief vector.
    """

    h_internal = full_internal_energy(agent_beliefs, belief_weights)
    h_social = 0.0
    if include_social:
        h_social = social_energy(
            agent_idx,
            agent_beliefs,
            all_beliefs,
            agent_adjacency,
            discussion_beliefs,
            normalize_neighbor_influence=normalize_neighbor_influence,
        )
    return beta_internal * h_internal + beta_social * h_social


def agent_energy(
    agent_idx: int,
    all_beliefs: np.ndarray,
    belief_weights: np.ndarray,
    agent_adjacency: np.ndarray,
    allowed_values: Mapping[int, Iterable[float]] | None,
    discussion_beliefs: Sequence[int],
    *,
    beta_internal: float,
    beta_social: float,
    include_social: bool = True,
    normalize_neighbor_influence: bool = False,
) -> dict[str, float]:
    """Return energy components and total pressure for one agent.

    This is a convenience wrapper for inspection/debugging: it validates the
    discussion belief indices, computes H_internal and H_social, and returns
    them together with D under the supplied beta parameters.

    Args:
        agent_idx: Index of the selected agent.
        all_beliefs: Belief matrix for all agents.
        belief_weights: Square matrix of weights between belief dimensions.
        agent_adjacency: Square social adjacency matrix between agents.
        allowed_values: Optional mapping from belief index to allowed states.
            Used here for validating discussion belief indices.
        discussion_beliefs: Belief indices included in the current discussion.
        beta_internal: Strength of internal consistency pressure.
        beta_social: Strength of neighbor social pressure.
        normalize_neighbor_influence: Whether to normalize neighbor weights
            before aggregating social influence.

    Returns:
        Dictionary with H_internal, H_social, and total pressure D.
    """

    n_beliefs = np.asarray(belief_weights).shape[0]
    allowed = normalize_allowed_values(n_beliefs, allowed_values)
    missing = [idx for idx in discussion_beliefs if idx not in allowed]
    if missing:
        raise ValueError(f"Unknown belief indices in discussion_beliefs: {missing}")

    agent_beliefs = np.asarray(all_beliefs, dtype=float)[agent_idx]
    h_internal = full_internal_energy(agent_beliefs, belief_weights)
    h_social = 0.0
    if include_social:
        h_social = social_energy(
            agent_idx,
            agent_beliefs,
            np.asarray(all_beliefs, dtype=float),
            agent_adjacency,
            discussion_beliefs,
            normalize_neighbor_influence=normalize_neighbor_influence,
        )
    return {
        "H_internal": h_internal,
        "H_social": h_social,
        "D": beta_internal * h_internal + beta_social * h_social,
    }


def logistic_weight(delta_pressure: np.ndarray) -> np.ndarray:
    """Compute article-style unnormalized transition weights stably.

    The R code uses 1 / (1 + exp(D_new - D_old)). This helper implements
    the same logistic expression without numerical overflow for large
    positive or negative pressure differences.

    Args:
        delta_pressure: Difference D_new - D_old for one or more candidate
            states.

    Returns:
        Unnormalized logistic weights for candidate states.
    """

    delta = np.asarray(delta_pressure, dtype=float)
    out = np.empty_like(delta)
    positive = delta >= 0
    out[positive] = np.exp(-delta[positive]) / (1.0 + np.exp(-delta[positive]))
    out[~positive] = 1.0 / (1.0 + np.exp(delta[~positive]))
    return out


def transition_probabilities(
    agent_idx: int,
    belief_idx: int,
    all_beliefs: np.ndarray,
    belief_weights: np.ndarray,
    agent_adjacency: np.ndarray,
    allowed_values: Mapping[int, np.ndarray],
    discussion_beliefs: Sequence[int],
    *,
    beta_internal: float,
    beta_social: float,
    include_social: bool = True,
    normalize_neighbor_influence: bool = False,
) -> np.ndarray:
    """Compute normalized probabilities for one agent's one belief update.

    The function tries every allowed value for belief_idx, recomputes total
    pressure D for that candidate state, converts pressure differences into
    logistic weights, and normalizes those weights so they sum to one.

    Args:
        agent_idx: Index of the selected agent.
        belief_idx: Belief dimension being updated.
        all_beliefs: Belief matrix for all agents.
        belief_weights: Square matrix of weights between belief dimensions.
        agent_adjacency: Square social adjacency matrix between agents.
        allowed_values: Mapping from belief index to allowed candidate states.
        discussion_beliefs: Belief indices included in the current discussion.
        beta_internal: Strength of internal consistency pressure.
        beta_social: Strength of neighbor social pressure.
        normalize_neighbor_influence: Whether to normalize neighbor weights
            before aggregating social influence.

    Returns:
        Probability vector over allowed_values[belief_idx].
    """

    current_agent_beliefs = all_beliefs[agent_idx].copy()
    current_pressure = pressure(
        agent_idx,
        current_agent_beliefs,
        all_beliefs,
        belief_weights,
        agent_adjacency,
        discussion_beliefs,
        beta_internal=beta_internal,
        beta_social=beta_social,
        include_social=include_social,
        normalize_neighbor_influence=normalize_neighbor_influence,
    )

    candidate_pressures = []
    for candidate in allowed_values[belief_idx]:
        candidate_agent_beliefs = current_agent_beliefs.copy()
        candidate_agent_beliefs[belief_idx] = candidate
        candidate_pressures.append(
            pressure(
                agent_idx,
                candidate_agent_beliefs,
                all_beliefs,
                belief_weights,
                agent_adjacency,
                discussion_beliefs,
                beta_internal=beta_internal,
                beta_social=beta_social,
                include_social=include_social,
                normalize_neighbor_influence=normalize_neighbor_influence,
            )
        )

    unnormalized = logistic_weight(np.asarray(candidate_pressures) - current_pressure)
    total = float(unnormalized.sum())
    if total == 0 or not np.isfinite(total):
        return np.full_like(unnormalized, 1.0 / unnormalized.size)
    return unnormalized / total


def joint_transition_probabilities(
    agent_idx: int,
    all_beliefs: np.ndarray,
    belief_weights: np.ndarray,
    agent_adjacency: np.ndarray,
    allowed_values: Mapping[int, np.ndarray],
    discussion_beliefs: Sequence[int],
    *,
    beta_internal: float,
    beta_social: float,
    normalize_neighbor_influence: bool = False,
) -> tuple[list[tuple[float, ...]], np.ndarray]:
    """Compute probabilities over all joint candidate states for Q."""

    discussion = np.asarray(discussion_beliefs, dtype=int)
    current_agent_beliefs = all_beliefs[agent_idx].copy()
    current_pressure = pressure(
        agent_idx,
        current_agent_beliefs,
        all_beliefs,
        belief_weights,
        agent_adjacency,
        discussion,
        beta_internal=beta_internal,
        beta_social=beta_social,
        normalize_neighbor_influence=normalize_neighbor_influence,
    )

    value_grids = [allowed_values[int(belief_idx)] for belief_idx in discussion]
    candidates = [tuple(float(v) for v in values) for values in product(*value_grids)]
    candidate_pressures = np.empty(len(candidates), dtype=float)

    for candidate_idx, candidate_values in enumerate(candidates):
        candidate_agent_beliefs = current_agent_beliefs.copy()
        candidate_agent_beliefs[discussion] = candidate_values
        candidate_pressures[candidate_idx] = pressure(
            agent_idx,
            candidate_agent_beliefs,
            all_beliefs,
            belief_weights,
            agent_adjacency,
            discussion,
            beta_internal=beta_internal,
            beta_social=beta_social,
            normalize_neighbor_influence=normalize_neighbor_influence,
        )

    unnormalized = logistic_weight(candidate_pressures - current_pressure)
    total = float(unnormalized.sum())
    if total == 0 or not np.isfinite(total):
        return candidates, np.full_like(unnormalized, 1.0 / unnormalized.size)
    return candidates, unnormalized / total


def sample_discussion_beliefs(
    rng: np.random.Generator,
    n_beliefs: int,
    discussion_size: int | tuple[int, int],
) -> np.ndarray:
    """Randomly choose which belief dimensions are discussed this step.

    discussion_size may be a fixed integer or an inclusive (low, high) range.
    Sampling is without replacement, so each discussed belief appears once.

    Args:
        rng: NumPy random generator.
        n_beliefs: Total number of belief dimensions per agent.
        discussion_size: Fixed discussion size or inclusive range of sizes.

    Returns:
        Array of belief indices selected for the current discussion.
    """

    if isinstance(discussion_size, tuple):
        low, high = discussion_size
        size = int(rng.integers(low, high + 1))
    else:
        size = int(discussion_size)

    if size <= 0:
        raise ValueError("discussion_size must be positive.")
    size = min(size, n_beliefs)
    return rng.choice(n_beliefs, size=size, replace=False)


def sample_topic_belief(
    rng: np.random.Generator,
    n_beliefs: int,
) -> int:
    """Randomly choose one belief dimension as the current discussion topic."""

    return int(rng.integers(n_beliefs))


def initialize_beliefs(
    n_agents: int,
    n_beliefs: int,
    allowed_values: Mapping[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an initial belief matrix from per-belief allowed values.

    Args:
        n_agents: Number of agents in the social network.
        n_beliefs: Number of belief dimensions per agent.
        allowed_values: Mapping from belief index to allowed states.
        rng: NumPy random generator.

    Returns:
        Initial belief matrix with shape (n_agents, n_beliefs).
    """

    beliefs = np.empty((n_agents, n_beliefs), dtype=float)
    for belief_idx in range(n_beliefs):
        beliefs[:, belief_idx] = rng.choice(allowed_values[belief_idx], size=n_agents)
    return beliefs


def run_exchange(
    belief_weights: np.ndarray | Sequence[Sequence[float]],
    agent_adjacency: np.ndarray | Sequence[Sequence[float]],
    allowed_values: Mapping[int, Iterable[float]] | None = None,
    initial_beliefs: np.ndarray | Sequence[Sequence[float]] | None = None,
    config: ExchangeConfig = ExchangeConfig(),
    *,
    normalize_neighbor_influence: bool = False,
    return_history: bool = False,
    show_progress: bool | None = None,
) -> tuple[ig.Graph, list[dict[str, object]]]:
    """Run the direct social opinion-exchange process.

    Inputs are the belief-weight matrix, agent-adjacency matrix, optional
    allowed values and optional initial belief matrix. At each step, a random
    agent and exactly one belief topic are selected. If the topic is focal,
    transition probabilities use full internal energy plus social impact for
    that topic. If it is not focal, probabilities use only full internal
    energy. show_progress overrides config.show_progress when passed. The
    returned igraph graph stores final belief vectors in vertex attribute
    "beliefs".

    Args:
        belief_weights: Square matrix of weights between belief dimensions.
            Positive weights reward alignment; negative weights reward
            opposition.
        agent_adjacency: Square social adjacency matrix between agents.
            Nonzero values indicate neighbor influence strength.
        allowed_values: Optional mapping from belief index to allowed states.
            Missing entries default to seven values from -1 to 1.
        initial_beliefs: Optional initial belief matrix. If omitted, beliefs
            are sampled from allowed_values.
        config: ExchangeConfig object with beta values, discussion size,
            number of steps, progress setting, and random seed.
        normalize_neighbor_influence: Whether to normalize neighbor weights
            before aggregating social influence.
        return_history: Whether to store per-step update records.
        show_progress: Optional direct override for config.show_progress.

    Returns:
        Pair (graph, history), where graph is the final igraph.Graph and
        history is either empty or a list of per-step records.
    """

    weights = np.asarray(belief_weights, dtype=float)
    adjacency = np.asarray(agent_adjacency, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("belief_weights must be a square matrix.")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("agent_adjacency must be a square matrix.")

    validate_single_topic_config(config)
    rng = np.random.default_rng(config.seed)
    n_beliefs = weights.shape[0]
    n_agents = adjacency.shape[0]
    allowed = normalize_allowed_values(n_beliefs, allowed_values)
    focal_mask = normalize_focal_beliefs(n_beliefs, config.focal_beliefs)

    if initial_beliefs is None:
        beliefs = initialize_beliefs(n_agents, n_beliefs, allowed, rng)
    else:
        beliefs = np.asarray(initial_beliefs, dtype=float).copy()
        if beliefs.shape != (n_agents, n_beliefs):
            raise ValueError("initial_beliefs must have shape (n_agents, n_beliefs).")

    graph = build_agent_graph(adjacency, beliefs)
    history: list[dict[str, object]] = []

    progress_enabled = config.show_progress if show_progress is None else show_progress
    steps = range(config.n_steps)
    if progress_enabled:
        steps = tqdm(steps, desc="Opinion exchange", leave=False)

    for step in steps:
        agent_idx = int(rng.integers(n_agents))
        topic = sample_topic_belief(rng, n_beliefs)
        is_focal = bool(focal_mask[topic])

        probs = transition_probabilities(
            agent_idx,
            topic,
            beliefs,
            weights,
            adjacency,
            allowed,
            [topic],
            beta_internal=config.beta_internal,
            beta_social=config.beta_social,
            include_social=is_focal,
            normalize_neighbor_influence=normalize_neighbor_influence,
        )
        beliefs[agent_idx, topic] = rng.choice(allowed[topic], p=probs)

        if return_history:
            history.append(
                {
                    "step": step,
                    "agent": agent_idx,
                    "topic": topic,
                    "is_focal": is_focal,
                    "discussion_beliefs": [topic],
                    "beliefs": beliefs[agent_idx].copy(),
                }
            )

    set_graph_belief_matrix(graph, beliefs)
    return graph, history


cdef inline uint64_t _rng_next(uint64_t *state) noexcept nogil:
    cdef uint64_t x = state[0]
    if x == 0:
        x = <uint64_t>0x9E3779B97F4A7C15
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state[0] = x
    return x * <uint64_t>2685821657736338717


cdef inline int _rng_int(uint64_t *state, int high) noexcept nogil:
    if high <= 1:
        return 0
    return <int>(_rng_next(state) % <uint64_t>high)


cdef inline double _rng_unit(uint64_t *state) noexcept nogil:
    return ((_rng_next(state) >> 11) * (1.0 / 9007199254740992.0))


cdef inline double _logistic_weight_c(double delta) noexcept nogil:
    if delta >= 0.0:
        return exp(-delta) / (1.0 + exp(-delta))
    return 1.0 / (1.0 + exp(delta))


cdef dict _allowed_values_to_dense(int n_beliefs, object allowed_values):
    """Convert possibly ragged allowed-value dict to dense values + counts."""
    cdef dict normalized = normalize_allowed_values(n_beliefs, allowed_values)
    cdef int belief_idx
    cdef int max_count = 0
    cdef cnp.ndarray values_array

    for belief_idx in range(n_beliefs):
        values_array = np.asarray(normalized[belief_idx], dtype=np.float64)
        if values_array.size > max_count:
            max_count = values_array.size

    cdef cnp.ndarray dense = np.empty((n_beliefs, max_count), dtype=np.float64)
    cdef cnp.ndarray counts = np.empty(n_beliefs, dtype=np.int32)
    dense.fill(0.0)
    for belief_idx in range(n_beliefs):
        values_array = np.asarray(normalized[belief_idx], dtype=np.float64)
        counts[belief_idx] = values_array.size
        dense[belief_idx, :values_array.size] = values_array

    return {"values": dense, "counts": counts}


def adjacency_to_csr(agent_adjacency):
    """Convert dense/sparse adjacency input to CSR arrays used by fast sparse code.

    Accepted forms:
        - scipy-like sparse matrix with .tocsr(), .indptr, .indices, .data.
        - CSR-like object with .indptr, .indices, .data, and .shape.
        - tuple/list (indptr, indices, data, shape).
        - dense square matrix.

    Returns:
        Dictionary with int32 indptr/indices, float64 data, and shape.
    """
    cdef object csr
    cdef cnp.ndarray dense
    cdef int n_agents
    cdef int i
    cdef int j
    cdef int nnz = 0
    cdef int pos = 0
    cdef cnp.ndarray indptr
    cdef cnp.ndarray indices
    cdef cnp.ndarray data

    if isinstance(agent_adjacency, (tuple, list)) and len(agent_adjacency) == 4:
        indptr = np.ascontiguousarray(agent_adjacency[0], dtype=np.int32)
        indices = np.ascontiguousarray(agent_adjacency[1], dtype=np.int32)
        data = np.ascontiguousarray(agent_adjacency[2], dtype=np.float64)
        shape = tuple(agent_adjacency[3])
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("CSR shape must be square.")
        return {"indptr": indptr, "indices": indices, "data": data, "shape": shape}

    if hasattr(agent_adjacency, "tocsr"):
        csr = agent_adjacency.tocsr()
        indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
        data = np.ascontiguousarray(csr.data, dtype=np.float64)
        shape = tuple(csr.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("agent_adjacency must be square.")
        return {"indptr": indptr, "indices": indices, "data": data, "shape": shape}

    if all(hasattr(agent_adjacency, attr) for attr in ("indptr", "indices", "data", "shape")):
        indptr = np.ascontiguousarray(agent_adjacency.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(agent_adjacency.indices, dtype=np.int32)
        data = np.ascontiguousarray(agent_adjacency.data, dtype=np.float64)
        shape = tuple(agent_adjacency.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("agent_adjacency must be square.")
        return {"indptr": indptr, "indices": indices, "data": data, "shape": shape}

    dense = np.ascontiguousarray(agent_adjacency, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("agent_adjacency must be a square matrix.")

    n_agents = <int>dense.shape[0]
    for i in range(n_agents):
        for j in range(n_agents):
            if dense[i, j] != 0.0:
                nnz += 1

    indptr = np.empty(n_agents + 1, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    indptr[0] = 0
    for i in range(n_agents):
        for j in range(n_agents):
            if dense[i, j] != 0.0:
                indices[pos] = j
                data[pos] = dense[i, j]
                pos += 1
        indptr[i + 1] = pos

    return {"indptr": indptr, "indices": indices, "data": data, "shape": (n_agents, n_agents)}


def build_agent_graph_from_csr(indptr, indices, data, initial_beliefs, *, directed=True):
    """Create an igraph agent network from CSR adjacency arrays and beliefs."""
    cdef cnp.ndarray indptr_np = np.ascontiguousarray(indptr, dtype=np.int32)
    cdef cnp.ndarray indices_np = np.ascontiguousarray(indices, dtype=np.int32)
    cdef cnp.ndarray data_np = np.ascontiguousarray(data, dtype=np.float64)
    cdef cnp.ndarray beliefs = np.asarray(initial_beliefs, dtype=float)
    cdef int n_agents = <int>(indptr_np.shape[0] - 1)
    cdef int i
    cdef int p
    cdef int j
    cdef list edges = []
    cdef list weights = []

    if beliefs.ndim != 2 or beliefs.shape[0] != n_agents:
        raise ValueError("initial_beliefs must have one row per agent.")

    for i in range(n_agents):
        for p in range(indptr_np[i], indptr_np[i + 1]):
            j = <int>indices_np[p]
            if i == j:
                continue
            if directed or i < j:
                edges.append((i, j))
                weights.append(float(data_np[p]))

    graph = ig.Graph(n=n_agents, edges=edges, directed=directed)
    graph.vs["agent_id"] = list(range(n_agents))
    graph.vs["beliefs"] = [row.tolist() for row in beliefs]
    if weights:
        graph.es["weight"] = weights
    return graph


cdef inline double _agent_belief_value(
    double[:, ::1] beliefs,
    int agent_idx,
    int belief_idx,
    int candidate_idx,
    double candidate_value,
) noexcept nogil:
    if belief_idx == candidate_idx:
        return candidate_value
    return beliefs[agent_idx, belief_idx]


cdef inline double _agent_belief_value_joint(
    double[:, ::1] beliefs,
    int agent_idx,
    int belief_idx,
    int[::1] discussion,
    int discussion_size,
    double[::1] combo_values,
) noexcept nogil:
    cdef int pos
    for pos in range(discussion_size):
        if discussion[pos] == belief_idx:
            return combo_values[pos]
    return beliefs[agent_idx, belief_idx]


cdef double _pressure_joint_dense_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    double[:, ::1] agent_adjacency,
    int[::1] discussion,
    int discussion_size,
    double[::1] combo_values,
    double beta_internal,
    double beta_social,
    bint normalize_neighbor_influence,
) noexcept nogil:
    cdef int n_agents = beliefs.shape[0]
    cdef int n_beliefs = beliefs.shape[1]
    cdef int a
    cdef int b
    cdef int pos
    cdef int neighbor
    cdef int selected_idx
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double value_a
    cdef double value_b
    cdef double selected_value
    cdef double neighbor_signal
    cdef double denom = 0.0
    cdef double weight

    for a in range(n_beliefs):
        value_a = _agent_belief_value_joint(beliefs, agent_idx, a, discussion, discussion_size, combo_values)
        for b in range(n_beliefs):
            value_b = _agent_belief_value_joint(beliefs, agent_idx, b, discussion, discussion_size, combo_values)
            h_internal -= belief_weights[a, b] * value_a * value_b

    if normalize_neighbor_influence:
        for neighbor in range(n_agents):
            if neighbor != agent_idx:
                denom += fabs(agent_adjacency[agent_idx, neighbor])
        if denom == 0.0:
            denom = 1.0
    else:
        denom = 1.0

    for pos in range(discussion_size):
        selected_idx = discussion[pos]
        selected_value = combo_values[pos]
        neighbor_signal = 0.0
        for neighbor in range(n_agents):
            if neighbor == agent_idx:
                continue
            weight = agent_adjacency[agent_idx, neighbor]
            if weight != 0.0:
                neighbor_signal += weight * beliefs[neighbor, selected_idx]
        h_social -= selected_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


cdef double _pressure_joint_sparse_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    int[::1] indptr,
    int[::1] indices,
    double[::1] edge_data,
    int[::1] discussion,
    int discussion_size,
    double[::1] combo_values,
    double beta_internal,
    double beta_social,
    bint normalize_neighbor_influence,
) noexcept nogil:
    cdef int n_beliefs = beliefs.shape[1]
    cdef int a
    cdef int b
    cdef int pos
    cdef int p
    cdef int neighbor
    cdef int selected_idx
    cdef int row_start = indptr[agent_idx]
    cdef int row_end = indptr[agent_idx + 1]
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double value_a
    cdef double value_b
    cdef double selected_value
    cdef double neighbor_signal
    cdef double denom = 0.0
    cdef double weight

    for a in range(n_beliefs):
        value_a = _agent_belief_value_joint(beliefs, agent_idx, a, discussion, discussion_size, combo_values)
        for b in range(n_beliefs):
            value_b = _agent_belief_value_joint(beliefs, agent_idx, b, discussion, discussion_size, combo_values)
            h_internal -= belief_weights[a, b] * value_a * value_b

    if normalize_neighbor_influence:
        for p in range(row_start, row_end):
            neighbor = indices[p]
            if neighbor != agent_idx:
                denom += fabs(edge_data[p])
        if denom == 0.0:
            denom = 1.0
    else:
        denom = 1.0

    for pos in range(discussion_size):
        selected_idx = discussion[pos]
        selected_value = combo_values[pos]
        neighbor_signal = 0.0
        for p in range(row_start, row_end):
            neighbor = indices[p]
            if neighbor == agent_idx:
                continue
            weight = edge_data[p]
            neighbor_signal += weight * beliefs[neighbor, selected_idx]
        h_social -= selected_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


cdef double _pressure_topic_dense_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    double[:, ::1] agent_adjacency,
    int topic_idx,
    double candidate_value,
    double beta_internal,
    double beta_social,
    bint include_social,
    bint normalize_neighbor_influence,
) noexcept nogil:
    cdef int n_agents = beliefs.shape[0]
    cdef int n_beliefs = beliefs.shape[1]
    cdef int a
    cdef int b
    cdef int neighbor
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double value_a
    cdef double value_b
    cdef double neighbor_signal = 0.0
    cdef double denom = 0.0
    cdef double weight

    for a in range(n_beliefs):
        value_a = _agent_belief_value(beliefs, agent_idx, a, topic_idx, candidate_value)
        for b in range(n_beliefs):
            value_b = _agent_belief_value(beliefs, agent_idx, b, topic_idx, candidate_value)
            h_internal -= belief_weights[a, b] * value_a * value_b

    if include_social:
        if normalize_neighbor_influence:
            for neighbor in range(n_agents):
                if neighbor != agent_idx:
                    denom += fabs(agent_adjacency[agent_idx, neighbor])
            if denom == 0.0:
                denom = 1.0
        else:
            denom = 1.0

        for neighbor in range(n_agents):
            if neighbor == agent_idx:
                continue
            weight = agent_adjacency[agent_idx, neighbor]
            if weight != 0.0:
                neighbor_signal += weight * beliefs[neighbor, topic_idx]
        h_social -= candidate_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


cdef double _pressure_topic_sparse_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    int[::1] indptr,
    int[::1] indices,
    double[::1] edge_data,
    int topic_idx,
    double candidate_value,
    double beta_internal,
    double beta_social,
    bint include_social,
    bint normalize_neighbor_influence,
) noexcept nogil:
    cdef int n_beliefs = beliefs.shape[1]
    cdef int a
    cdef int b
    cdef int p
    cdef int neighbor
    cdef int row_start = indptr[agent_idx]
    cdef int row_end = indptr[agent_idx + 1]
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double value_a
    cdef double value_b
    cdef double neighbor_signal = 0.0
    cdef double denom = 0.0
    cdef double weight

    for a in range(n_beliefs):
        value_a = _agent_belief_value(beliefs, agent_idx, a, topic_idx, candidate_value)
        for b in range(n_beliefs):
            value_b = _agent_belief_value(beliefs, agent_idx, b, topic_idx, candidate_value)
            h_internal -= belief_weights[a, b] * value_a * value_b

    if include_social:
        if normalize_neighbor_influence:
            for p in range(row_start, row_end):
                neighbor = indices[p]
                if neighbor != agent_idx:
                    denom += fabs(edge_data[p])
            if denom == 0.0:
                denom = 1.0
        else:
            denom = 1.0

        for p in range(row_start, row_end):
            neighbor = indices[p]
            if neighbor == agent_idx:
                continue
            weight = edge_data[p]
            neighbor_signal += weight * beliefs[neighbor, topic_idx]
        h_social -= candidate_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


cdef double _pressure_fast_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    double[:, ::1] agent_adjacency,
    int[::1] discussion,
    int discussion_size,
    double beta_internal,
    double beta_social,
    bint normalize_neighbor_influence,
    int candidate_idx,
    double candidate_value,
) noexcept nogil:
    cdef int n_agents = beliefs.shape[0]
    cdef int n_beliefs = beliefs.shape[1]
    cdef int pos
    cdef int j
    cdef int neighbor
    cdef int selected_idx
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double selected_value
    cdef double other_value
    cdef double neighbor_signal
    cdef double denom = 0.0
    cdef double weight

    if normalize_neighbor_influence:
        for neighbor in range(n_agents):
            if neighbor != agent_idx:
                denom += fabs(agent_adjacency[agent_idx, neighbor])
        if denom == 0.0:
            denom = 1.0
    else:
        denom = 1.0

    for pos in range(discussion_size):
        selected_idx = discussion[pos]
        selected_value = _agent_belief_value(beliefs, agent_idx, selected_idx, candidate_idx, candidate_value)

        for j in range(n_beliefs):
            other_value = _agent_belief_value(beliefs, agent_idx, j, candidate_idx, candidate_value)
            h_internal -= belief_weights[selected_idx, j] * selected_value * other_value

        neighbor_signal = 0.0
        for neighbor in range(n_agents):
            if neighbor == agent_idx:
                continue
            weight = agent_adjacency[agent_idx, neighbor]
            if weight != 0.0:
                neighbor_signal += weight * beliefs[neighbor, selected_idx]
        h_social -= selected_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


cdef double _pressure_sparse_c(
    int agent_idx,
    double[:, ::1] beliefs,
    double[:, ::1] belief_weights,
    int[::1] indptr,
    int[::1] indices,
    double[::1] edge_data,
    int[::1] discussion,
    int discussion_size,
    double beta_internal,
    double beta_social,
    bint normalize_neighbor_influence,
    int candidate_idx,
    double candidate_value,
) noexcept nogil:
    cdef int n_beliefs = beliefs.shape[1]
    cdef int pos
    cdef int j
    cdef int p
    cdef int neighbor
    cdef int selected_idx
    cdef int row_start = indptr[agent_idx]
    cdef int row_end = indptr[agent_idx + 1]
    cdef double h_internal = 0.0
    cdef double h_social = 0.0
    cdef double selected_value
    cdef double other_value
    cdef double neighbor_signal
    cdef double denom = 0.0
    cdef double weight

    if normalize_neighbor_influence:
        for p in range(row_start, row_end):
            neighbor = indices[p]
            if neighbor != agent_idx:
                denom += fabs(edge_data[p])
        if denom == 0.0:
            denom = 1.0
    else:
        denom = 1.0

    for pos in range(discussion_size):
        selected_idx = discussion[pos]
        selected_value = _agent_belief_value(beliefs, agent_idx, selected_idx, candidate_idx, candidate_value)

        for j in range(n_beliefs):
            other_value = _agent_belief_value(beliefs, agent_idx, j, candidate_idx, candidate_value)
            h_internal -= belief_weights[selected_idx, j] * selected_value * other_value

        neighbor_signal = 0.0
        for p in range(row_start, row_end):
            neighbor = indices[p]
            if neighbor == agent_idx:
                continue
            weight = edge_data[p]
            neighbor_signal += weight * beliefs[neighbor, selected_idx]
        h_social -= selected_value * (neighbor_signal / denom)

    return beta_internal * h_internal + beta_social * h_social


def run_exchange_fast(
    belief_weights,
    agent_adjacency,
    allowed_values=None,
    initial_beliefs=None,
    config=ExchangeConfig(),
    *,
    normalize_neighbor_influence=False,
    return_history=False,
    show_progress=None,
):
    """Optimized Cython runner for the direct social belief model.

    This function keeps the same model semantics as run_exchange, but moves
    the hot update loop into typed Cython code. It uses a lightweight C-level
    xorshift RNG, so trajectories with the same seed are reproducible within
    this fast function but not bit-identical to the Python/NumPy runner.
    """
    cdef cnp.ndarray weights_np = np.ascontiguousarray(belief_weights, dtype=np.float64)
    cdef cnp.ndarray adjacency_np = np.ascontiguousarray(agent_adjacency, dtype=np.float64)
    cdef int n_beliefs
    cdef int n_agents
    cdef dict allowed_dense
    cdef cnp.ndarray allowed_np
    cdef cnp.ndarray counts_np
    cdef cnp.ndarray focal_mask_np
    cdef cnp.ndarray beliefs_np
    cdef double[:, ::1] weights_view
    cdef double[:, ::1] adjacency_view
    cdef double[:, ::1] beliefs_view
    cdef double[:, ::1] allowed_view
    cdef int[::1] counts_view
    cdef unsigned char[::1] focal_mask_view
    cdef cnp.ndarray discussion_np
    cdef cnp.ndarray pool_np
    cdef cnp.ndarray probs_np
    cdef cnp.ndarray combo_values_np
    cdef int[::1] discussion_view
    cdef int[::1] pool_view
    cdef double[::1] probs_view
    cdef double[::1] combo_values_view
    cdef int step
    cdef int agent_idx
    cdef int discussion_size
    cdef int topic_idx
    cdef int low
    cdef int high
    cdef int m
    cdef int r
    cdef int tmp
    cdef int belief_idx
    cdef int candidate_pos
    cdef int candidate_count
    cdef int chosen_pos
    cdef int combo_count
    cdef int combo_id
    cdef int combo_tmp
    cdef int chosen_combo_id
    cdef int value_pos
    cdef double current_pressure
    cdef double candidate_pressure
    cdef double candidate_value
    cdef double total_weight
    cdef double draw
    cdef double cumulative
    cdef double beta_internal = float(config.beta_internal)
    cdef double beta_social = float(config.beta_social)
    cdef int n_steps = int(config.n_steps)
    cdef bint normalize = bool(normalize_neighbor_influence)
    cdef bint topic_is_focal
    cdef bint progress_enabled = bool(config.show_progress if show_progress is None else show_progress)
    cdef uint64_t rng_state
    cdef object progress = None
    cdef int progress_chunk = 1024
    cdef int progress_since_update = 0
    cdef list history = []

    if weights_np.ndim != 2 or weights_np.shape[0] != weights_np.shape[1]:
        raise ValueError("belief_weights must be a square matrix.")
    if adjacency_np.ndim != 2 or adjacency_np.shape[0] != adjacency_np.shape[1]:
        raise ValueError("agent_adjacency must be a square matrix.")

    n_beliefs = <int>weights_np.shape[0]
    n_agents = <int>adjacency_np.shape[0]

    validate_single_topic_config(config)
    allowed_dense = _allowed_values_to_dense(n_beliefs, allowed_values)
    allowed_np = np.ascontiguousarray(allowed_dense["values"], dtype=np.float64)
    counts_np = np.ascontiguousarray(allowed_dense["counts"], dtype=np.int32)
    focal_mask_np = np.ascontiguousarray(
        normalize_focal_beliefs(n_beliefs, getattr(config, "focal_beliefs", None)),
        dtype=np.uint8,
    )

    if initial_beliefs is None:
        beliefs_np = np.empty((n_agents, n_beliefs), dtype=np.float64)
    else:
        beliefs_np = np.ascontiguousarray(initial_beliefs, dtype=np.float64).copy()
        if beliefs_np.shape[0] != n_agents or beliefs_np.shape[1] != n_beliefs:
            raise ValueError("initial_beliefs must have shape (n_agents, n_beliefs).")

    weights_view = weights_np
    adjacency_view = adjacency_np
    beliefs_view = beliefs_np
    allowed_view = allowed_np
    counts_view = counts_np
    focal_mask_view = focal_mask_np

    discussion_np = np.empty(n_beliefs, dtype=np.int32)
    pool_np = np.empty(n_beliefs, dtype=np.int32)
    probs_np = np.empty(allowed_np.shape[1], dtype=np.float64)
    combo_values_np = np.empty(n_beliefs, dtype=np.float64)
    discussion_view = discussion_np
    pool_view = pool_np
    probs_view = probs_np
    combo_values_view = combo_values_np

    rng_state = <uint64_t>(config.seed if config.seed is not None else 88172645463393265)
    if rng_state == 0:
        rng_state = 88172645463393265

    if initial_beliefs is None:
        for belief_idx in range(n_beliefs):
            candidate_count = counts_view[belief_idx]
            for agent_idx in range(n_agents):
                beliefs_view[agent_idx, belief_idx] = allowed_view[belief_idx, _rng_int(&rng_state, candidate_count)]

    discussion_size = 1

    if progress_enabled:
        progress = tqdm(total=n_steps, desc="Opinion exchange fast", leave=False)

    try:
        for step in range(n_steps):
            agent_idx = _rng_int(&rng_state, n_agents)
            topic_idx = _rng_int(&rng_state, n_beliefs)
            topic_is_focal = focal_mask_view[topic_idx] != 0
            candidate_count = counts_view[topic_idx]

            current_pressure = _pressure_topic_dense_c(
                agent_idx,
                beliefs_view,
                weights_view,
                adjacency_view,
                topic_idx,
                beliefs_view[agent_idx, topic_idx],
                beta_internal,
                beta_social,
                topic_is_focal,
                normalize,
            )

            total_weight = 0.0
            for candidate_pos in range(candidate_count):
                candidate_value = allowed_view[topic_idx, candidate_pos]
                candidate_pressure = _pressure_topic_dense_c(
                    agent_idx,
                    beliefs_view,
                    weights_view,
                    adjacency_view,
                    topic_idx,
                    candidate_value,
                    beta_internal,
                    beta_social,
                    topic_is_focal,
                    normalize,
                )
                probs_view[candidate_pos] = _logistic_weight_c(candidate_pressure - current_pressure)
                total_weight += probs_view[candidate_pos]

            if total_weight <= 0.0 or not isfinite(total_weight):
                chosen_pos = _rng_int(&rng_state, candidate_count)
            else:
                draw = _rng_unit(&rng_state) * total_weight
                cumulative = 0.0
                chosen_pos = candidate_count - 1
                for candidate_pos in range(candidate_count):
                    cumulative += probs_view[candidate_pos]
                    if draw <= cumulative:
                        chosen_pos = candidate_pos
                        break

            beliefs_view[agent_idx, topic_idx] = allowed_view[topic_idx, chosen_pos]

            if return_history:
                history.append(
                    {
                        "step": step,
                        "agent": agent_idx,
                        "topic": topic_idx,
                        "is_focal": bool(topic_is_focal),
                        "discussion_beliefs": [topic_idx],
                        "beliefs": np.asarray(beliefs_np[agent_idx]).copy(),
                    }
                )

            if progress is not None:
                progress_since_update += 1
                if progress_since_update >= progress_chunk:
                    progress.update(progress_since_update)
                    progress_since_update = 0
    finally:
        if progress is not None:
            if progress_since_update:
                progress.update(progress_since_update)
            progress.close()

    return build_agent_graph(adjacency_np, beliefs_np), history


def run_exchange_fast_sparse(
    belief_weights,
    agent_adjacency,
    allowed_values=None,
    initial_beliefs=None,
    config=ExchangeConfig(),
    *,
    normalize_neighbor_influence=False,
    return_history=False,
    show_progress=None,
    directed_graph=True,
):
    """Optimized Cython runner for sparse social networks.

    The social graph is stored as CSR arrays, so every update scans only the
    selected agent's real neighbors instead of all agents. This is the right
    runner for large sparse graphs, e.g. 1000 agents with small average degree.
    """
    cdef cnp.ndarray weights_np = np.ascontiguousarray(belief_weights, dtype=np.float64)
    cdef dict csr = adjacency_to_csr(agent_adjacency)
    cdef cnp.ndarray indptr_np = np.ascontiguousarray(csr["indptr"], dtype=np.int32)
    cdef cnp.ndarray indices_np = np.ascontiguousarray(csr["indices"], dtype=np.int32)
    cdef cnp.ndarray data_np = np.ascontiguousarray(csr["data"], dtype=np.float64)
    cdef tuple shape = tuple(csr["shape"])
    cdef int n_beliefs
    cdef int n_agents
    cdef dict allowed_dense
    cdef cnp.ndarray allowed_np
    cdef cnp.ndarray counts_np
    cdef cnp.ndarray focal_mask_np
    cdef cnp.ndarray beliefs_np
    cdef double[:, ::1] weights_view
    cdef double[:, ::1] beliefs_view
    cdef double[:, ::1] allowed_view
    cdef int[::1] counts_view
    cdef unsigned char[::1] focal_mask_view
    cdef int[::1] indptr_view
    cdef int[::1] indices_view
    cdef double[::1] data_view
    cdef cnp.ndarray discussion_np
    cdef cnp.ndarray pool_np
    cdef cnp.ndarray probs_np
    cdef cnp.ndarray combo_values_np
    cdef int[::1] discussion_view
    cdef int[::1] pool_view
    cdef double[::1] probs_view
    cdef double[::1] combo_values_view
    cdef int step
    cdef int agent_idx
    cdef int discussion_size
    cdef int topic_idx
    cdef int low
    cdef int high
    cdef int m
    cdef int r
    cdef int tmp
    cdef int belief_idx
    cdef int candidate_pos
    cdef int candidate_count
    cdef int chosen_pos
    cdef int combo_count
    cdef int combo_id
    cdef int combo_tmp
    cdef int chosen_combo_id
    cdef int value_pos
    cdef double current_pressure
    cdef double candidate_pressure
    cdef double candidate_value
    cdef double total_weight
    cdef double draw
    cdef double cumulative
    cdef double beta_internal = float(config.beta_internal)
    cdef double beta_social = float(config.beta_social)
    cdef int n_steps = int(config.n_steps)
    cdef bint normalize = bool(normalize_neighbor_influence)
    cdef bint topic_is_focal
    cdef bint progress_enabled = bool(config.show_progress if show_progress is None else show_progress)
    cdef uint64_t rng_state
    cdef object progress = None
    cdef int progress_chunk = 1024
    cdef int progress_since_update = 0
    cdef list history = []

    if weights_np.ndim != 2 or weights_np.shape[0] != weights_np.shape[1]:
        raise ValueError("belief_weights must be a square matrix.")
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("agent_adjacency must be square.")

    n_beliefs = <int>weights_np.shape[0]
    n_agents = <int>shape[0]
    if indptr_np.shape[0] != n_agents + 1:
        raise ValueError("CSR indptr length must be n_agents + 1.")
    if indices_np.shape[0] != data_np.shape[0]:
        raise ValueError("CSR indices and data must have equal length.")

    validate_single_topic_config(config)
    allowed_dense = _allowed_values_to_dense(n_beliefs, allowed_values)
    allowed_np = np.ascontiguousarray(allowed_dense["values"], dtype=np.float64)
    counts_np = np.ascontiguousarray(allowed_dense["counts"], dtype=np.int32)
    focal_mask_np = np.ascontiguousarray(
        normalize_focal_beliefs(n_beliefs, getattr(config, "focal_beliefs", None)),
        dtype=np.uint8,
    )

    if initial_beliefs is None:
        beliefs_np = np.empty((n_agents, n_beliefs), dtype=np.float64)
    else:
        beliefs_np = np.ascontiguousarray(initial_beliefs, dtype=np.float64).copy()
        if beliefs_np.shape[0] != n_agents or beliefs_np.shape[1] != n_beliefs:
            raise ValueError("initial_beliefs must have shape (n_agents, n_beliefs).")

    weights_view = weights_np
    beliefs_view = beliefs_np
    allowed_view = allowed_np
    counts_view = counts_np
    focal_mask_view = focal_mask_np
    indptr_view = indptr_np
    indices_view = indices_np
    data_view = data_np

    discussion_np = np.empty(n_beliefs, dtype=np.int32)
    pool_np = np.empty(n_beliefs, dtype=np.int32)
    probs_np = np.empty(allowed_np.shape[1], dtype=np.float64)
    combo_values_np = np.empty(n_beliefs, dtype=np.float64)
    discussion_view = discussion_np
    pool_view = pool_np
    probs_view = probs_np
    combo_values_view = combo_values_np

    rng_state = <uint64_t>(config.seed if config.seed is not None else 88172645463393265)
    if rng_state == 0:
        rng_state = 88172645463393265

    if initial_beliefs is None:
        for belief_idx in range(n_beliefs):
            candidate_count = counts_view[belief_idx]
            for agent_idx in range(n_agents):
                beliefs_view[agent_idx, belief_idx] = allowed_view[belief_idx, _rng_int(&rng_state, candidate_count)]

    discussion_size = 1

    if progress_enabled:
        progress = tqdm(total=n_steps, desc="Opinion exchange sparse", leave=False)

    try:
        for step in range(n_steps):
            agent_idx = _rng_int(&rng_state, n_agents)
            topic_idx = _rng_int(&rng_state, n_beliefs)
            topic_is_focal = focal_mask_view[topic_idx] != 0
            candidate_count = counts_view[topic_idx]

            current_pressure = _pressure_topic_sparse_c(
                agent_idx,
                beliefs_view,
                weights_view,
                indptr_view,
                indices_view,
                data_view,
                topic_idx,
                beliefs_view[agent_idx, topic_idx],
                beta_internal,
                beta_social,
                topic_is_focal,
                normalize,
            )

            total_weight = 0.0
            for candidate_pos in range(candidate_count):
                candidate_value = allowed_view[topic_idx, candidate_pos]
                candidate_pressure = _pressure_topic_sparse_c(
                    agent_idx,
                    beliefs_view,
                    weights_view,
                    indptr_view,
                    indices_view,
                    data_view,
                    topic_idx,
                    candidate_value,
                    beta_internal,
                    beta_social,
                    topic_is_focal,
                    normalize,
                )
                probs_view[candidate_pos] = _logistic_weight_c(candidate_pressure - current_pressure)
                total_weight += probs_view[candidate_pos]

            if total_weight <= 0.0 or not isfinite(total_weight):
                chosen_pos = _rng_int(&rng_state, candidate_count)
            else:
                draw = _rng_unit(&rng_state) * total_weight
                cumulative = 0.0
                chosen_pos = candidate_count - 1
                for candidate_pos in range(candidate_count):
                    cumulative += probs_view[candidate_pos]
                    if draw <= cumulative:
                        chosen_pos = candidate_pos
                        break

            beliefs_view[agent_idx, topic_idx] = allowed_view[topic_idx, chosen_pos]

            if return_history:
                history.append(
                    {
                        "step": step,
                        "agent": agent_idx,
                        "topic": topic_idx,
                        "is_focal": bool(topic_is_focal),
                        "discussion_beliefs": [topic_idx],
                        "beliefs": np.asarray(beliefs_np[agent_idx]).copy(),
                    }
                )

            if progress is not None:
                progress_since_update += 1
                if progress_since_update >= progress_chunk:
                    progress.update(progress_since_update)
                    progress_since_update = 0
    finally:
        if progress is not None:
            if progress_since_update:
                progress.update(progress_since_update)
            progress.close()

    return build_agent_graph_from_csr(
        indptr_np,
        indices_np,
        data_np,
        beliefs_np,
        directed=bool(directed_graph),
    ), history


if __name__ == "__main__":
    belief_weights = np.full((5, 5), 0.4)
    np.fill_diagonal(belief_weights, 0.0)

    agent_adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )

    graph, _ = run_exchange(
        belief_weights,
        agent_adjacency,
        config=ExchangeConfig(beta_internal=1.0, beta_social=1.5, focal_beliefs=[0, 2, 4], n_steps=100, seed=42),
    )
    print(np.asarray(graph.vs["beliefs"]))


