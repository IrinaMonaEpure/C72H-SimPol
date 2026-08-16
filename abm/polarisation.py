from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
from warnings import warn
from zipfile import BadZipFile

import numpy as np
from scipy.stats import kendalltau


SnapshotSelector = Literal["first", "last"] | int


@dataclass(frozen=True)
class EffectiveDimensionalityResult:
    """Effective dimensionality diagnostics for one opinion matrix."""

    d_eff: float
    pc1_share: float
    eigenvalues: np.ndarray
    proportions: np.ndarray
    kendall_matrix: np.ndarray
    active_beliefs: np.ndarray
    negative_eigenvalue_mass: float


@dataclass(frozen=True)
class EffectiveDimensionalityTimeSeries:
    """Effective dimensionality diagnostics for selected snapshots."""

    steps: np.ndarray
    snapshot_indices: np.ndarray
    d_eff: np.ndarray
    pc1_share: np.ndarray
    active_counts: np.ndarray
    negative_eigenvalue_mass: np.ndarray
    results: tuple[EffectiveDimensionalityResult, ...] | None = None


def effective_dimensionality(
    X: np.ndarray | Sequence[Sequence[float]],
    belief_indices: Sequence[int] | None = None,
) -> EffectiveDimensionalityResult:
    """Compute entropy effective rank from a Kendall tau-b correlation matrix.

    Args:
        X: Opinion matrix with shape ``(n_agents, n_beliefs)``.
        belief_indices: Optional subset of belief columns to include.

    Returns:
        EffectiveDimensionalityResult with the effective dimensionality,
        first-component share, eigen diagnostics, Kendall matrix, and active
        belief indices. Constant belief columns are excluded because their
        Kendall correlations are undefined and they do not carry disagreement.
    """

    opinions = _as_2d_float_matrix(X, name="X")
    selected = _normalize_belief_indices(opinions.shape[1], belief_indices)
    active = _active_belief_indices(opinions, selected)
    n_active = active.size

    if n_active == 0:
        return EffectiveDimensionalityResult(
            d_eff=0.0,
            pc1_share=np.nan,
            eigenvalues=np.asarray([], dtype=float),
            proportions=np.asarray([], dtype=float),
            kendall_matrix=np.empty((0, 0), dtype=float),
            active_beliefs=active,
            negative_eigenvalue_mass=0.0,
        )

    if n_active == 1:
        return EffectiveDimensionalityResult(
            d_eff=1.0,
            pc1_share=1.0,
            eigenvalues=np.asarray([1.0], dtype=float),
            proportions=np.asarray([1.0], dtype=float),
            kendall_matrix=np.asarray([[1.0]], dtype=float),
            active_beliefs=active,
            negative_eigenvalue_mass=0.0,
        )

    kendall_matrix = _kendall_matrix(opinions, active)
    eigenvalues = np.linalg.eigvalsh((kendall_matrix + kendall_matrix.T) / 2.0)[::-1]
    negative_eigenvalue_mass = float(np.abs(eigenvalues[eigenvalues < 0.0]).sum())
    clipped = np.clip(eigenvalues, 0.0, None)
    total = float(clipped.sum())

    if total <= 0.0:
        proportions = np.zeros_like(clipped)
        d_eff = 0.0
        pc1_share = np.nan
    else:
        proportions = clipped / total
        positive = proportions[proportions > 0.0]
        d_eff = float(np.exp(-np.sum(positive * np.log(positive))))
        pc1_share = float(proportions[0])

    return EffectiveDimensionalityResult(
        d_eff=d_eff,
        pc1_share=pc1_share,
        eigenvalues=eigenvalues,
        proportions=proportions,
        kendall_matrix=kendall_matrix,
        active_beliefs=active,
        negative_eigenvalue_mass=negative_eigenvalue_mass,
    )


def effective_dimensionality_from_graph(
    graph: object,
    belief_attr: str = "beliefs",
    belief_indices: Sequence[int] | None = None,
) -> EffectiveDimensionalityResult:
    """Compute effective dimensionality from an igraph vertex belief attribute."""

    try:
        beliefs = graph.vs[belief_attr]  # type: ignore[attr-defined]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f'graph must expose vertex attribute "{belief_attr}"') from exc

    return effective_dimensionality(np.asarray(beliefs, dtype=float), belief_indices=belief_indices)


def effective_dimensionality_over_time(
    belief_history: np.ndarray | Sequence[Sequence[Sequence[float]]],
    steps: Sequence[int] | np.ndarray | None = None,
    snapshot_indices: Sequence[int] | None = None,
    keep_matrices: bool = False,
    belief_indices: Sequence[int] | None = None,
) -> EffectiveDimensionalityTimeSeries:
    """Compute effective dimensionality for selected history snapshots.

    Args:
        belief_history: Array with shape ``(n_snapshots, n_agents, n_beliefs)``.
        steps: Optional step labels for snapshots. Defaults to snapshot indices.
        snapshot_indices: Optional subset of snapshot positions. Defaults to
            all snapshots.
        keep_matrices: Whether to keep full per-snapshot result objects.
        belief_indices: Optional subset of belief columns to include.
    """

    history = np.asarray(belief_history, dtype=float)
    if history.ndim != 3:
        raise ValueError("belief_history must have shape (n_snapshots, n_agents, n_beliefs).")

    n_snapshots = history.shape[0]
    selected = _normalize_snapshot_indices(n_snapshots, snapshot_indices)

    if steps is None:
        all_steps = np.arange(n_snapshots, dtype=np.int64)
    else:
        all_steps = np.asarray(steps)
        if all_steps.ndim != 1 or all_steps.size != n_snapshots:
            raise ValueError("steps must be a 1D array with one value per snapshot.")

    results = tuple(
        effective_dimensionality(history[int(idx)], belief_indices=belief_indices)
        for idx in selected
    )

    return EffectiveDimensionalityTimeSeries(
        steps=all_steps[selected],
        snapshot_indices=selected,
        d_eff=np.asarray([result.d_eff for result in results], dtype=float),
        pc1_share=np.asarray([result.pc1_share for result in results], dtype=float),
        active_counts=np.asarray([result.active_beliefs.size for result in results], dtype=int),
        negative_eigenvalue_mass=np.asarray(
            [result.negative_eigenvalue_mass for result in results],
            dtype=float,
        ),
        results=results if keep_matrices else None,
    )


def analyse_experiment_effective_dimensionality(
    results_dir: str | Path,
    snapshots: Sequence[SnapshotSelector] = ("first", "last"),
    on_error: Literal["warn", "raise", "ignore"] = "warn",
    belief_indices: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Analyse effective dimensionality for every ``.npz`` run in a directory.

    Each returned row contains run metadata, the requested snapshot label,
    step, effective dimensionality, first-component share, active-belief count,
    and eigenvalue diagnostics. Bad or incomplete files are skipped when
    ``on_error`` is ``"warn"`` or ``"ignore"``.
    """

    if on_error not in {"warn", "raise", "ignore"}:
        raise ValueError('on_error must be one of "warn", "raise", or "ignore".')

    directory = Path(results_dir)
    files = sorted(directory.glob("*.npz"))
    rows: list[dict[str, object]] = []

    for file_path in files:
        try:
            with file_path.open("rb") as file_handle:
                with np.load(file_handle, allow_pickle=False) as data:
                    if "belief_history" not in data:
                        raise KeyError('missing "belief_history" array')

                    history = data["belief_history"]
                    steps = data["steps"] if "steps" in data else np.arange(history.shape[0])
                    run_metadata = _extract_run_metadata(data)

                    for snapshot in snapshots:
                        snapshot_index = _resolve_snapshot_selector(snapshot, history.shape[0])
                        result = effective_dimensionality(
                            history[snapshot_index],
                            belief_indices=belief_indices,
                        )
                        rows.append(
                            {
                                "file": file_path.name,
                                "snapshot": snapshot,
                                "snapshot_index": int(snapshot_index),
                                "step": int(steps[snapshot_index]),
                                "d_eff": result.d_eff,
                                "pc1_share": result.pc1_share,
                                "active_beliefs": result.active_beliefs.copy(),
                                "active_belief_count": int(result.active_beliefs.size),
                                "eigenvalues": result.eigenvalues.copy(),
                                "proportions": result.proportions.copy(),
                                "negative_eigenvalue_mass": result.negative_eigenvalue_mass,
                                **run_metadata,
                            }
                        )
        except (OSError, BadZipFile, KeyError, ValueError) as exc:
            _handle_experiment_error(file_path, exc, on_error)

    return rows


def save_effective_dimensionality_history_by_params(
    results_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    parameter_keys: Sequence[str] = ("beta_internal", "beta_external"),
    snapshot_indices: Sequence[int] | None = None,
    on_error: Literal["warn", "raise", "ignore"] = "warn",
    belief_indices: Sequence[int] | None = None,
    output_format: Literal["auto", "json", "pickle"] = "auto",
) -> dict[tuple[object, ...], tuple[tuple[int, float, float], ...]]:
    """Compute and save full-history dimensionality grouped by run parameters.

    The returned Python object is a dictionary like::

        {
            (0.5, 2.0): (
                (0, 19.95, 0.056),
                (40000, 18.90, 0.120),
                ...
            ),
        }

    Args:
        results_dir: Directory with ``.npz`` experiment runs.
        output_path: File to write. Defaults to
            ``results_dir / "effective_dimensionality_by_params.json"``.
        parameter_keys: Scalar arrays in each ``.npz`` used as dictionary key.
        snapshot_indices: Optional subset of history snapshots. Defaults to
            every snapshot in each file.
        on_error: How to handle bad/incomplete files.
        belief_indices: Optional subset of belief columns to include.
        output_format: ``"json"``, ``"pickle"``, or ``"auto"``. Auto infers
            pickle from ``.pkl``/``.pickle`` and JSON otherwise.

    Returns:
        Dictionary keyed by parameter tuples. JSON output is written as a
        readable list of runs because JSON cannot represent tuple keys.
    """

    if on_error not in {"warn", "raise", "ignore"}:
        raise ValueError('on_error must be one of "warn", "raise", or "ignore".')
    if output_format not in {"auto", "json", "pickle"}:
        raise ValueError('output_format must be one of "auto", "json", or "pickle".')

    directory = Path(results_dir)
    destination = (
        directory / "effective_dimensionality_by_params.json"
        if output_path is None
        else Path(output_path)
    )
    grouped: dict[tuple[object, ...], tuple[tuple[int, float, float], ...]] = {}

    for file_path in sorted(directory.glob("*.npz")):
        try:
            with file_path.open("rb") as file_handle:
                with np.load(file_handle, allow_pickle=False) as data:
                    if "belief_history" not in data:
                        raise KeyError('missing "belief_history" array')

                    history = data["belief_history"]
                    steps = data["steps"] if "steps" in data else np.arange(history.shape[0])
                    key = _extract_parameter_key(data, parameter_keys)
                    series = effective_dimensionality_over_time(
                        history,
                        steps=steps,
                        snapshot_indices=snapshot_indices,
                        keep_matrices=False,
                        belief_indices=belief_indices,
                    )

                    if key in grouped:
                        warn(
                            f"Overwriting duplicate parameter key {key} from {file_path}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    grouped[key] = tuple(
                        (
                            int(step),
                            float(d_eff),
                            float(pc1_share),
                        )
                        for step, d_eff, pc1_share in zip(
                            series.steps,
                            series.d_eff,
                            series.pc1_share,
                        )
                    )
        except (OSError, BadZipFile, KeyError, ValueError) as exc:
            _handle_experiment_error(file_path, exc, on_error)

    _write_grouped_history(
        grouped,
        destination,
        parameter_keys=parameter_keys,
        output_format=output_format,
    )

    return grouped


def _as_2d_float_matrix(
    values: np.ndarray | Sequence[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_agents, n_beliefs).")
    return array


def _normalize_belief_indices(
    n_beliefs: int,
    belief_indices: Sequence[int] | None,
) -> np.ndarray:
    if belief_indices is None:
        return np.arange(n_beliefs, dtype=int)

    indices = np.asarray(list(belief_indices), dtype=int)
    if indices.ndim != 1:
        raise ValueError("belief_indices must be a 1D sequence of column indices.")
    if np.any(indices < 0) or np.any(indices >= n_beliefs):
        raise ValueError("belief_indices contains an out-of-range column index.")
    if np.unique(indices).size != indices.size:
        raise ValueError("belief_indices must not contain duplicates.")
    return indices


def _active_belief_indices(opinions: np.ndarray, selected: np.ndarray) -> np.ndarray:
    active: list[int] = []
    for belief_idx in selected:
        column = opinions[:, int(belief_idx)]
        finite = column[np.isfinite(column)]
        if finite.size > 1 and np.unique(finite).size > 1:
            active.append(int(belief_idx))
    return np.asarray(active, dtype=int)


def _kendall_matrix(opinions: np.ndarray, active: np.ndarray) -> np.ndarray:
    matrix = np.eye(active.size, dtype=float)
    for left_pos, left_idx in enumerate(active):
        for right_pos in range(left_pos + 1, active.size):
            right_idx = active[right_pos]
            tau = kendalltau(
                opinions[:, int(left_idx)],
                opinions[:, int(right_idx)],
                nan_policy="omit",
            ).statistic
            matrix[left_pos, right_pos] = matrix[right_pos, left_pos] = (
                0.0 if np.isnan(tau) else float(tau)
            )
    return matrix


def _normalize_snapshot_indices(
    n_snapshots: int,
    snapshot_indices: Sequence[int] | None,
) -> np.ndarray:
    if snapshot_indices is None:
        return np.arange(n_snapshots, dtype=int)

    indices = np.asarray(list(snapshot_indices), dtype=int)
    if indices.ndim != 1:
        raise ValueError("snapshot_indices must be a 1D sequence of indices.")
    indices = np.where(indices < 0, indices + n_snapshots, indices)
    if np.any(indices < 0) or np.any(indices >= n_snapshots):
        raise ValueError("snapshot_indices contains an out-of-range snapshot index.")
    return indices


def _resolve_snapshot_selector(snapshot: SnapshotSelector, n_snapshots: int) -> int:
    if snapshot == "first":
        return 0
    if snapshot == "last":
        return n_snapshots - 1
    if isinstance(snapshot, (int, np.integer)):
        index = int(snapshot)
        if index < 0:
            index += n_snapshots
        if index < 0 or index >= n_snapshots:
            raise ValueError(f"snapshot index {snapshot} is out of range.")
        return index
    raise ValueError('snapshots must contain "first", "last", or integer indices.')


def _extract_run_metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("beta_internal", "beta_external"):
        if key in data:
            metadata[key] = float(np.asarray(data[key]))
    if "seed" in data:
        metadata["seed"] = int(np.asarray(data["seed"]))
    if "focal_beliefs" in data:
        metadata["focal_beliefs"] = np.asarray(data["focal_beliefs"], dtype=int).copy()
    return metadata


def _extract_parameter_key(
    data: np.lib.npyio.NpzFile,
    parameter_keys: Sequence[str],
) -> tuple[object, ...]:
    values: list[object] = []
    for key in parameter_keys:
        if key not in data:
            raise KeyError(f'missing parameter array "{key}"')
        values.append(_scalar_npz_value(data[key]))
    return tuple(values)


def _scalar_npz_value(value: np.ndarray) -> object:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("parameter arrays must be scalar.")
    item = array.item()
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _write_grouped_history(
    grouped: dict[tuple[object, ...], tuple[tuple[int, float, float], ...]],
    destination: Path,
    *,
    parameter_keys: Sequence[str],
    output_format: Literal["auto", "json", "pickle"],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(destination, output_format)

    if resolved_format == "pickle":
        with destination.open("wb") as file_handle:
            pickle.dump(grouped, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    payload = _grouped_history_to_json_payload(grouped, parameter_keys)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _resolve_output_format(
    destination: Path,
    output_format: Literal["auto", "json", "pickle"],
) -> Literal["json", "pickle"]:
    if output_format != "auto":
        return output_format
    if destination.suffix.lower() in {".pkl", ".pickle"}:
        return "pickle"
    return "json"


def _grouped_history_to_json_payload(
    grouped: dict[tuple[object, ...], tuple[tuple[int, float, float], ...]],
    parameter_keys: Sequence[str],
) -> dict[str, object]:
    runs = []
    for key in sorted(grouped):
        parameters = {
            parameter_name: key[idx]
            for idx, parameter_name in enumerate(parameter_keys)
        }
        runs.append(
            {
                "key": list(key),
                "parameters": parameters,
                "series": [
                    {
                        "step": int(step),
                        "d_eff": _json_number(d_eff),
                        "pc1": _json_number(pc1_share),
                    }
                    for step, d_eff, pc1_share in grouped[key]
                ],
            }
        )

    return {
        "parameter_keys": list(parameter_keys),
        "value_columns": ["step", "d_eff", "pc1"],
        "runs": runs,
    }


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _handle_experiment_error(
    file_path: Path,
    exc: Exception,
    on_error: Literal["warn", "raise", "ignore"],
) -> None:
    if on_error == "raise":
        raise exc
    if on_error == "warn":
        warn(f"Skipping {file_path}: {exc}", RuntimeWarning, stacklevel=2)
