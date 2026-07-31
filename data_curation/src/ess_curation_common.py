"""Shared utilities for ESS Round 4 and Round 8 data curation.

This module contains only round-independent operations. The principal
analysis sample retains all adult respondents regardless of belief
missingness, while preserving a flag for the historical CCA missingness
criterion. Round-specific variables, item mappings, coding directions,
and expected values belong in ``ess4_config.py`` and ``ess8_config.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the data_curation project root.

    The function works when called from either the project root or one of its
    immediate subfolders, such as ``notebooks`` or ``scripts``.
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (
            (candidate / "data").is_dir()
            and (candidate / "notebooks").is_dir()
            and (candidate / "src").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the data_curation project root. Expected folders "
        "'data', 'notebooks', and 'src'."
    )


def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str = "dataframe",
) -> None:
    """Raise a clear error when required columns are absent."""
    required = list(columns)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns in {context}: {missing}")


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert a series to numeric; non-numeric values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def valid_range(
    series: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    """Keep only values within an inclusive substantive range."""
    if maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum")

    numeric = to_numeric_clean(series)
    return numeric.where(numeric.between(minimum, maximum))


def scale_keep(
    series: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    """Rescale valid responses to 0–1 while preserving direction."""
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    numeric = valid_range(series, minimum, maximum)
    return (numeric - minimum) / (maximum - minimum)


def scale_reverse(
    series: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    """Reverse-code valid responses and rescale them to 0–1."""
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    numeric = valid_range(series, minimum, maximum)
    return (maximum - numeric) / (maximum - minimum)



def recode_values(
    series: pd.Series,
    value_mapping: Mapping[float, float],
    *,
    unmapped_to_nan: bool = True,
) -> pd.Series:
    """Apply an explicit item-level value mapping.

    Parameters
    ----------
    series:
        Raw ESS item responses.
    value_mapping:
        Mapping from original values to recoded values.
    unmapped_to_nan:
        When ``True`` (the default), any numeric value absent from the mapping
        becomes missing. This matches special ESS recodes such as Round 4
        ``txearn``. When ``False``, values absent from the mapping are retained.
    """
    numeric = to_numeric_clean(series)

    if unmapped_to_nan:
        return numeric.map(dict(value_mapping))

    return numeric.replace(dict(value_mapping))


def force_values_missing(
    series: pd.Series,
    values: Iterable[float],
) -> pd.Series:
    """Convert explicitly listed response values to missing values."""
    numeric = to_numeric_clean(series)
    forced_missing = list(values)
    if not forced_missing:
        return numeric
    return numeric.mask(numeric.isin(forced_missing))


def clean_and_scale_item(
    series: pd.Series,
    direction: str,
    minimum: float,
    maximum: float,
    *,
    value_mapping: Mapping[float, float] | None = None,
    values_forced_missing: Iterable[float] = (),
) -> pd.Series:
    """Apply special recoding, invalid-value cleaning, and 0-1 scaling.

    The order is intentional: explicit value mappings are applied first, then
    any listed values are forced to missing, and finally the substantive range
    and coding direction are enforced.
    """
    cleaned = to_numeric_clean(series)

    if value_mapping is not None:
        cleaned = recode_values(
            cleaned,
            value_mapping,
            unmapped_to_nan=True,
        )

    cleaned = force_values_missing(cleaned, values_forced_missing)

    if direction == "keep":
        return scale_keep(cleaned, minimum, maximum)
    if direction == "reverse":
        return scale_reverse(cleaned, minimum, maximum)

    raise ValueError(
        f"Unknown coding direction {direction!r}; expected 'keep' or 'reverse'."
    )


def construct_belief_variables(
    dataframe: pd.DataFrame,
    belief_map: Mapping[str, Sequence[str]],
    item_coding: Mapping[str, tuple[str, float, float]],
    *,
    item_value_recoding: Mapping[str, Mapping[float, float]] | None = None,
    item_values_forced_missing: Mapping[str, Iterable[float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct round-specific 0-1 belief variables from raw ESS items.

    Returns
    -------
    beliefs:
        One column per constructed belief, in ``belief_map`` order.
    coded_items:
        The cleaned and 0-1-scaled constituent ESS items. This second output is
        useful for transparent notebook checks and validation.

    Multi-item beliefs use :func:`strict_row_mean`, so all constituent items
    must be available for the constructed belief to be non-missing.
    """
    value_recoding = item_value_recoding or {}
    forced_missing = item_values_forced_missing or {}

    required_items = tuple(
        dict.fromkeys(
            item
            for items in belief_map.values()
            for item in items
        )
    )
    require_columns(dataframe, required_items, context="raw belief input")

    missing_coding = [item for item in required_items if item not in item_coding]
    if missing_coding:
        raise KeyError(
            "Missing item-coding specifications for raw belief items: "
            f"{missing_coding}"
        )

    coded_items = pd.DataFrame(index=dataframe.index)
    for item in required_items:
        direction, minimum, maximum = item_coding[item]
        coded_items[item] = clean_and_scale_item(
            dataframe[item],
            direction,
            minimum,
            maximum,
            value_mapping=value_recoding.get(item),
            values_forced_missing=forced_missing.get(item, ()),
        )

    beliefs = pd.DataFrame(index=dataframe.index)
    for belief_name, items in belief_map.items():
        item_names = list(items)
        if not item_names:
            raise ValueError(
                f"Belief {belief_name!r} has no constituent ESS items."
            )
        if len(item_names) == 1:
            beliefs[belief_name] = coded_items[item_names[0]]
        else:
            beliefs[belief_name] = strict_row_mean(coded_items, item_names)

    return beliefs, coded_items

def strict_row_mean(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    """Calculate a row mean only when every constituent item is available."""
    if not columns:
        raise ValueError("columns must contain at least one item")

    require_columns(dataframe, columns, context="strict_row_mean input")
    return dataframe.loc[:, list(columns)].mean(axis=1, skipna=False)


def add_belief_missingness(
    dataframe: pd.DataFrame,
    belief_columns: Sequence[str],
    *,
    cca_maximum_missing_beliefs: int | None = None,
) -> pd.DataFrame:
    """Add belief-missingness counts and, optionally, CCA eligibility.

    No respondent is removed by this function.

    Parameters
    ----------
    dataframe:
        Respondent-level dataframe containing the constructed beliefs.
    belief_columns:
        Constructed belief variables used to calculate missingness.
    cca_maximum_missing_beliefs:
        Historical CCA threshold. When supplied, the Boolean column
        ``cca_missingness_eligible`` is added and indicates whether a
        respondent has no more than this number of missing beliefs.
    """
    require_columns(dataframe, belief_columns, context="belief dataframe")

    if (
        cca_maximum_missing_beliefs is not None
        and cca_maximum_missing_beliefs < 0
    ):
        raise ValueError(
            "cca_maximum_missing_beliefs must be non-negative or None"
        )

    result = dataframe.copy()
    result["n_belief_missing"] = (
        result.loc[:, belief_columns].isna().sum(axis=1)
    )
    result["n_belief_available"] = (
        len(belief_columns) - result["n_belief_missing"]
    )

    if cca_maximum_missing_beliefs is not None:
        result["cca_missingness_eligible"] = (
            result["n_belief_missing"]
            .le(cca_maximum_missing_beliefs)
            .astype(bool)
        )

    return result


def apply_analysis_sample_rule(
    dataframe: pd.DataFrame,
    belief_columns: Sequence[str],
    *,
    age_column: str = "agea",
    minimum_age: int = 18,
    cca_maximum_missing_beliefs: int = 2,
) -> pd.DataFrame:
    """Create the principal adult analysis sample without a missingness cutoff.

    Respondents are retained when their age is at least ``minimum_age`` or
    when age is missing. Respondents are *not* removed because of the number
    of missing constructed beliefs.

    The returned dataframe contains:

    - ``n_belief_missing``;
    - ``n_belief_available``; and
    - ``cca_missingness_eligible``, which records whether the respondent
      satisfies the historical CCA rule of no more than
      ``cca_maximum_missing_beliefs`` missing beliefs.

    Missing belief values remain missing and are not imputed.
    """
    require_columns(
        dataframe,
        [age_column, *belief_columns],
        context="analysis sample-rule input",
    )

    with_missingness = add_belief_missingness(
        dataframe,
        belief_columns,
        cca_maximum_missing_beliefs=cca_maximum_missing_beliefs,
    )

    age = to_numeric_clean(with_missingness[age_column])
    adult_or_missing_age = age.ge(minimum_age) | age.isna()

    return with_missingness.loc[adult_or_missing_age].copy()


def apply_cca_sample_rule(
    dataframe: pd.DataFrame,
    belief_columns: Sequence[str],
    *,
    age_column: str = "agea",
    minimum_age: int = 18,
    maximum_missing_beliefs: int = 2,
) -> pd.DataFrame:
    """Reproduce the historical CCA-restricted sample.

    This compatibility function retains respondents whose age is at least
    ``minimum_age`` or missing and who have no more than
    ``maximum_missing_beliefs`` missing constructed beliefs.

    It is retained only for reproducing and validating Van Noord et al.'s CCA
    sample. New principal datasets should use :func:`apply_analysis_sample_rule`
    instead.
    """
    analysis_sample = apply_analysis_sample_rule(
        dataframe,
        belief_columns,
        age_column=age_column,
        minimum_age=minimum_age,
        cca_maximum_missing_beliefs=maximum_missing_beliefs,
    )

    return analysis_sample.loc[
        analysis_sample["cca_missingness_eligible"]
    ].copy()


def split_weighted_and_unweighted(
    dataframe: pd.DataFrame,
    weight_columns: Sequence[str],
    *,
    insert_after: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create matching datasets without and with official ESS weights.

    The weighted version contains the same respondent-level data and inserts
    the supplied weight columns immediately after ``insert_after``. No belief
    values are transformed and no rows are replicated.
    """
    require_columns(dataframe, weight_columns, context="weighted source data")

    without_weights = dataframe.drop(columns=list(weight_columns)).copy()

    if insert_after not in without_weights.columns:
        raise KeyError(
            f"Cannot position weights: '{insert_after}' is not present in the "
            "without-weights dataframe."
        )

    base_columns = list(without_weights.columns)
    insertion_position = base_columns.index(insert_after) + 1
    with_weight_order = (
        base_columns[:insertion_position]
        + list(weight_columns)
        + base_columns[insertion_position:]
    )
    with_weights = dataframe.loc[:, with_weight_order].copy()

    validate_weight_split(
        without_weights,
        with_weights,
        weight_columns,
        require_complete_weights=False,
    )
    return without_weights, with_weights


def validate_weight_split(
    without_weights: pd.DataFrame,
    with_weights: pd.DataFrame,
    weight_columns: Sequence[str],
    *,
    require_complete_weights: bool = True,
) -> None:
    """Confirm that the two outputs differ only by weight columns."""
    require_columns(with_weights, weight_columns, context="with-weights data")

    if require_complete_weights and with_weights.loc[:, weight_columns].isna().any().any():
        missing_counts = (
            with_weights.loc[:, weight_columns].isna().sum().to_dict()
        )
        raise AssertionError(
            f"Missing values found in official weight columns: {missing_counts}"
        )

    assert_frame_equal(
        with_weights.drop(columns=list(weight_columns)).reset_index(drop=True),
        without_weights.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def summarise_beliefs(
    dataframe: pd.DataFrame,
    belief_columns: Sequence[str],
) -> pd.DataFrame:
    """Return N, mean, and sample SD for each constructed belief variable."""
    require_columns(dataframe, belief_columns, context="belief summary input")

    summary = (
        dataframe.loc[:, belief_columns]
        .agg(["count", "mean", "std"])
        .T
        .reset_index()
        .rename(
            columns={
                "index": "belief_variable",
                "count": "N_reproduced",
                "mean": "mean_reproduced",
                "std": "sd_reproduced",
            }
        )
    )
    summary["N_reproduced"] = summary["N_reproduced"].astype(int)
    return summary
