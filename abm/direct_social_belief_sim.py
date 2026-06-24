from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import igraph as ig
import numpy as np
from tqdm.auto import tqdm


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
    """Build a boolean mask of belief dimensions with active social discussion.

    Args:
        n_beliefs: Total number of belief dimensions per agent.
        focal_beliefs: Optional sequence of belief indices where social
            influence is active. If None, all dimensions are focal.

    Returns:
        Boolean mask with length n_beliefs. True means the selected topic uses
        internal plus social pressure; False means it uses only internal
        pressure.
    """

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
    """Reject legacy multi-topic discussion settings.

    Args:
        config: ExchangeConfig for the simulation run.

    Raises:
        ValueError: If discussion_size is not exactly 1.
    """

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

    Args:
        agent_beliefs: Current or candidate belief vector of one agent.
        belief_weights: Square matrix of weights between belief dimensions.

    Returns:
        Full internal energy H_internal for the whole belief vector.
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
        include_social: Whether to include social pressure. Non-focal topics
            set this to False and transition only from internal energy.
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
        include_social: Whether to include social pressure.
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
        include_social: Whether this topic receives social pressure.
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
    """Compute probabilities over all joint candidate states for Q.

    Q is treated as the set of degrees of freedom updated by this discussion.
    The function enumerates the Cartesian product of allowed values for all
    beliefs in Q, computes full internal energy plus social impact for every
    candidate belief vector, and normalizes article-style transition weights.

    Args:
        agent_idx: Index of the selected agent.
        all_beliefs: Belief matrix for all agents.
        belief_weights: Square matrix of weights between belief dimensions.
        agent_adjacency: Square social adjacency matrix between agents.
        allowed_values: Mapping from belief index to allowed candidate states.
        discussion_beliefs: Belief indices Q jointly updated this step.
        beta_internal: Strength of internal consistency pressure.
        beta_social: Strength of neighbor social pressure.
        normalize_neighbor_influence: Whether to normalize neighbor weights
            before aggregating social influence.

    Returns:
        Pair (candidates, probabilities). Each candidate is a tuple of values
        ordered like discussion_beliefs.
    """

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
    """Randomly choose one belief dimension as the current discussion topic.

    Args:
        rng: NumPy random generator.
        n_beliefs: Total number of belief dimensions per agent.

    Returns:
        One belief index selected for this step.
    """

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
        config: ExchangeConfig object with beta values, focal belief indices,
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
