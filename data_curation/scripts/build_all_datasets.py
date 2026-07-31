#!/usr/bin/env python3
"""Build and validate all ESS Round 4 and ESS Round 8 curated datasets.

Run this script from anywhere inside the project with:

    python3 scripts/build_all_datasets.py

The script:

1. reads the raw ESS4 and ESS8 CSV files;
2. constructs the round-specific metadata and belief variables;
3. applies the shared adult/max-two-missing-beliefs sample rule;
4. writes weighted and unweighted CSV files for both rounds;
5. validates ESS8 against Supplementary Table A2;
6. validates ESS4 respondent by respondent against Van Noord's df_ESS4.RData;
7. writes a descriptive comparison of the belief concepts shared by both rounds.

No raw or reference file is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ess4_config as ess4
from src import ess8_config as ess8
from src.ess_curation_common import (
    apply_cca_sample_rule,
    construct_belief_variables,
    require_columns,
    split_weighted_and_unweighted,
    summarise_beliefs,
    valid_range,
    validate_weight_split,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CROSS_ROUND_DESCRIPTIVES_FILENAME = (
    "ess4_ess8_shared_belief_descriptives.csv"
)
NUMERIC_TOLERANCE = 1e-12


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the ESS4 and ESS8 curated datasets and run all "
            "deterministic validation checks."
        )
    )
    parser.add_argument(
        "--skip-ess4-reference-validation",
        action="store_true",
        help=(
            "Build all four datasets without opening df_ESS4.RData. "
            "Use only when pyreadr or the reference RData file is unavailable."
        ),
    )
    return parser.parse_args()


def raw_data_path(config: ModuleType) -> Path:
    """Return the configured raw ESS CSV path."""
    return (
        PROJECT_ROOT
        / "data"
        / config.RAW_DATA_FOLDER
        / config.RAW_DATA_FILENAME
    )


def output_path(config: ModuleType, output_key: str) -> Path:
    """Return a configured processed-output path."""
    return PROCESSED_DIR / config.OUTPUT_FILENAMES[output_key]


def clean_metadata(
    raw: pd.DataFrame,
    config: ModuleType,
) -> pd.DataFrame:
    """Construct identifiers, demographics, voting metadata, and weights."""
    metadata = pd.DataFrame(index=raw.index)

    metadata["ess_row_id"] = np.arange(1, len(raw) + 1)
    metadata["idno"] = pd.to_numeric(
        raw["idno"],
        errors="coerce",
    ).astype("Int64")
    metadata["cntry"] = raw["cntry"].astype("string").str.strip()
    metadata["country_name"] = metadata["cntry"].map(config.COUNTRY_LABELS)
    metadata["ess_unique_id"] = (
        metadata["cntry"]
        + "_"
        + metadata["idno"].astype("string")
    )

    education_source = next(
        (
            column
            for column in ("eisced", "edulvla")
            if column in config.RAW_METADATA_COLUMNS
        ),
        None,
    )
    if education_source is None:
        raise KeyError(
            f"No education source variable found for {config.ROUND_LABEL}."
        )

    direct_metadata_columns = (
        "agea",
        "gndr",
        education_source,
        "hinctnta",
        "rlgblg",
        "blgetmg",
    )
    for column in direct_metadata_columns:
        minimum, maximum = config.METADATA_VALID_RANGES[column]
        metadata[column] = valid_range(raw[column], minimum, maximum)

    metadata["education_3cat"] = np.nan
    for category, source_values in config.EDUCATION_3CAT_MAP.items():
        metadata.loc[
            metadata[education_source].isin(source_values),
            "education_3cat",
        ] = category

    minimum, maximum = config.METADATA_VALID_RANGES["domicil"]
    domicil_clean = valid_range(raw["domicil"], minimum, maximum)
    metadata["urbanization"] = (
        config.URBANIZATION_REVERSE_CONSTANT - domicil_clean
    )

    if "vote_simple" in config.DEMOGRAPHIC_COLUMNS:
        vote_minimum, vote_maximum = config.METADATA_VALID_RANGES["vote"]
        metadata["vote_simple"] = valid_range(
            raw["vote"],
            vote_minimum,
            vote_maximum,
        )

    for column in config.WEIGHT_COLUMNS:
        metadata[column] = pd.to_numeric(raw[column], errors="coerce")

    metadata = metadata.loc[:, config.METADATA_COLUMNS_WITH_WEIGHTS]

    if metadata["country_name"].isna().any():
        unknown_codes = sorted(
            metadata.loc[
                metadata["country_name"].isna(),
                "cntry",
            ].dropna().unique()
        )
        raise AssertionError(
            f"Unmapped country codes in {config.ROUND_LABEL}: {unknown_codes}"
        )

    if metadata["ess_unique_id"].isna().any():
        raise AssertionError(
            f"Missing respondent identifiers in {config.ROUND_LABEL}."
        )

    if not metadata["ess_unique_id"].is_unique:
        raise AssertionError(
            f"Duplicate respondent identifiers in {config.ROUND_LABEL}."
        )

    missing_weights = (
        metadata.loc[:, config.WEIGHT_COLUMNS].isna().sum().to_dict()
    )
    if any(missing_weights.values()):
        raise AssertionError(
            f"Missing official weights in {config.ROUND_LABEL}: "
            f"{missing_weights}"
        )

    return metadata


def build_round(
    config: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one round's unweighted and weighted respondent-level datasets."""
    source_path = raw_data_path(config)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw {config.ROUND_LABEL} file not found: {source_path}"
        )

    print()
    print("=" * 72)
    print(f"Building {config.ROUND_LABEL}")
    print("=" * 72)
    print("Raw input:", source_path)

    raw = pd.read_csv(source_path, low_memory=False)

    require_columns(
        raw,
        config.REQUIRED_RAW_COLUMNS,
        context=f"raw {config.ROUND_LABEL} file",
    )

    if len(raw) != config.EXPECTED_RAW_N:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected {config.EXPECTED_RAW_N:,} "
            f"raw rows, found {len(raw):,}."
        )

    raw_country_count = raw["cntry"].nunique(dropna=True)
    if raw_country_count != config.EXPECTED_RAW_COUNTRIES:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected "
            f"{config.EXPECTED_RAW_COUNTRIES} raw countries, "
            f"found {raw_country_count}."
        )

    raw_weight_missing = (
        raw.loc[:, config.WEIGHT_COLUMNS].isna().sum().to_dict()
    )
    if any(raw_weight_missing.values()):
        raise AssertionError(
            f"{config.ROUND_LABEL}: missing raw ESS weights: "
            f"{raw_weight_missing}"
        )

    metadata = clean_metadata(raw, config)

    beliefs, coded_items = construct_belief_variables(
        raw,
        config.BELIEF_MAP,
        config.ITEM_CODING,
        item_value_recoding=getattr(
            config,
            "ITEM_VALUE_RECODING",
            {},
        ),
        item_values_forced_missing=getattr(
            config,
            "ITEM_VALUES_FORCED_MISSING",
            {},
        ),
    )

    if beliefs.shape != (len(raw), config.EXPECTED_BELIEF_COUNT):
        raise AssertionError(
            f"{config.ROUND_LABEL}: unexpected belief dataframe shape "
            f"{beliefs.shape}."
        )

    if list(beliefs.columns) != list(config.BELIEF_COLUMNS):
        raise AssertionError(
            f"{config.ROUND_LABEL}: belief-column order differs from config."
        )

    if list(coded_items.columns) != list(config.BELIEF_ITEM_COLUMNS):
        raise AssertionError(
            f"{config.ROUND_LABEL}: coded-item order differs from config."
        )

    observed_minimum = beliefs.min(skipna=True).min()
    observed_maximum = beliefs.max(skipna=True).max()
    if observed_minimum < 0.0 or observed_maximum > 1.0:
        raise AssertionError(
            f"{config.ROUND_LABEL}: belief values fall outside [0, 1]."
        )

    curated_all = pd.concat([metadata, beliefs], axis=1)
    curated_all = curated_all.sort_values(
        ["cntry", "idno"],
        kind="stable",
    ).reset_index(drop=True)

    cca_initial = apply_cca_sample_rule(
        curated_all,
        config.BELIEF_COLUMNS,
        age_column="agea",
        minimum_age=config.MINIMUM_AGE,
        maximum_missing_beliefs=config.MAXIMUM_MISSING_BELIEFS,
    ).reset_index(drop=True)

    if len(cca_initial) != config.EXPECTED_FINAL_N:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected {config.EXPECTED_FINAL_N:,} "
            f"final respondents, found {len(cca_initial):,}."
        )

    final_country_count = cca_initial["cntry"].nunique(dropna=True)
    if final_country_count != config.EXPECTED_FINAL_COUNTRIES:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected "
            f"{config.EXPECTED_FINAL_COUNTRIES} final countries, "
            f"found {final_country_count}."
        )

    if not cca_initial["n_belief_missing"].le(
        config.MAXIMUM_MISSING_BELIEFS
    ).all():
        raise AssertionError(
            f"{config.ROUND_LABEL}: final sample violates the missingness rule."
        )

    adult_or_missing_age = (
        cca_initial["agea"].ge(config.MINIMUM_AGE)
        | cca_initial["agea"].isna()
    )
    if not adult_or_missing_age.all():
        raise AssertionError(
            f"{config.ROUND_LABEL}: final sample violates the age rule."
        )

    without_weights, with_weights = split_weighted_and_unweighted(
        cca_initial,
        config.WEIGHT_COLUMNS,
        insert_after=config.WEIGHT_INSERT_AFTER,
    )

    without_weights = without_weights.loc[
        :, config.OUTPUT_COLUMNS_WITHOUT_WEIGHTS
    ].copy()
    with_weights = with_weights.loc[
        :, config.OUTPUT_COLUMNS_WITH_WEIGHTS
    ].copy()

    expected_without_shape = (
        config.EXPECTED_FINAL_N,
        len(config.OUTPUT_COLUMNS_WITHOUT_WEIGHTS),
    )
    expected_with_shape = (
        config.EXPECTED_FINAL_N,
        len(config.OUTPUT_COLUMNS_WITH_WEIGHTS),
    )

    if without_weights.shape != expected_without_shape:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected unweighted shape "
            f"{expected_without_shape}, found {without_weights.shape}."
        )

    if with_weights.shape != expected_with_shape:
        raise AssertionError(
            f"{config.ROUND_LABEL}: expected weighted shape "
            f"{expected_with_shape}, found {with_weights.shape}."
        )

    validate_weight_split(
        without_weights,
        with_weights,
        config.WEIGHT_COLUMNS,
        require_complete_weights=True,
    )

    print(f"Raw respondents: {len(raw):,}")
    print(f"Final respondents: {len(without_weights):,}")
    print(f"Countries: {final_country_count}")
    print(f"Belief variables: {len(config.BELIEF_COLUMNS)}")
    print(f"Without-weights shape: {without_weights.shape}")
    print(f"With-weights shape: {with_weights.shape}")

    return without_weights, with_weights


def validate_ess8_against_paper(
    ess8_without: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce Supplementary Table A2 validation for ESS8."""
    reproduced = summarise_beliefs(
        ess8_without,
        ess8.BELIEF_COLUMNS,
    )
    published = pd.DataFrame(
        ess8.PAPER_STATISTICS,
        columns=ess8.PAPER_STATISTICS_COLUMNS,
    )

    validation = published.merge(
        reproduced,
        on="belief_variable",
        how="left",
        validate="one_to_one",
    )
    validation["mean_reproduced_round2"] = (
        validation["mean_reproduced"].round(2)
    )
    validation["sd_reproduced_round2"] = (
        validation["sd_reproduced"].round(2)
    )
    validation["N_matches"] = (
        validation["N_paper"] == validation["N_reproduced"]
    )
    validation["mean_matches_round2"] = (
        validation["mean_paper"]
        == validation["mean_reproduced_round2"]
    )
    validation["sd_matches_round2"] = (
        validation["sd_paper"]
        == validation["sd_reproduced_round2"]
    )

    match_columns = (
        "N_matches",
        "mean_matches_round2",
        "sd_matches_round2",
    )
    if not validation.loc[:, match_columns].all().all():
        failed = validation.loc[
            ~validation.loc[:, match_columns].all(axis=1)
        ]
        raise AssertionError(
            "ESS8 does not reproduce Supplementary Table A2:\n"
            + failed.to_string(index=False)
        )

    print()
    print(
        "ESS8 validation passed: all 20 belief variables reproduce "
        "Supplementary Table A2."
    )
    return validation


def map_factor_to_numeric(
    series: pd.Series,
    *,
    label_mapping: dict[str, float],
    numeric_code_mapping: dict[float, float],
) -> pd.Series:
    """Map an R labelled factor or numeric code to the Python categories."""
    text_values = series.astype("string").str.strip()
    from_labels = text_values.map(label_mapping)

    numeric_values = pd.to_numeric(series, errors="coerce")
    from_codes = numeric_values.map(numeric_code_mapping)

    return pd.to_numeric(
        from_labels.fillna(from_codes),
        errors="coerce",
    )


def numeric_match_mask(
    python_values: pd.Series,
    reference_values: pd.Series,
    *,
    atol: float = NUMERIC_TOLERANCE,
) -> pd.Series:
    """Return element-wise equality, treating paired NaNs as equal."""
    python_numeric = pd.to_numeric(python_values, errors="coerce")
    reference_numeric = pd.to_numeric(reference_values, errors="coerce")

    both_missing = python_numeric.isna() & reference_numeric.isna()
    both_present = python_numeric.notna() & reference_numeric.notna()
    close = pd.Series(
        np.isclose(
            python_numeric.fillna(0.0),
            reference_numeric.fillna(0.0),
            rtol=0.0,
            atol=atol,
        ),
        index=python_numeric.index,
    )

    return both_missing | (both_present & close)


def validate_ess4_against_reference(
    ess4_without: pd.DataFrame,
) -> pd.DataFrame:
    """Compare ESS4 beliefs respondent by respondent with df_ESS4.RData."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required for ESS4 reference validation. "
            "Install it with `python3 -m pip install pyreadr`, or run "
            "with --skip-ess4-reference-validation."
        ) from exc

    reference_path = PROJECT_ROOT.joinpath(*ess4.REFERENCE_RDATA_PARTS)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"ESS4 reference RData file not found: {reference_path}"
        )

    r_objects = pyreadr.read_r(str(reference_path))
    if "df" in r_objects:
        reference_raw = r_objects["df"]
    elif len(r_objects) == 1:
        reference_raw = next(iter(r_objects.values()))
    else:
        raise KeyError(
            "Could not identify the ESS4 reference dataframe. "
            f"Objects found: {list(r_objects)}"
        )

    if not isinstance(reference_raw, pd.DataFrame):
        raise TypeError("The ESS4 RData object is not a dataframe.")

    if len(reference_raw) != ess4.EXPECTED_FINAL_N:
        raise AssertionError(
            f"ESS4 R reference has {len(reference_raw):,} rows; expected "
            f"{ess4.EXPECTED_FINAL_N:,}."
        )

    r_to_python_belief = {
        "lrscale": "left_right_identification",
        "gender_inequality": "gender_inequality",
        "anti_lgbt": "anti_lgbt",
        "euroscepticism": "euroscepticism",
        "anti_immigration": "anti_immigration",
        "anti_egalitarianism": "anti_egalitarianism",
        "benefits_eco": "benefits_harm_economy",
        "benefits_soc": "benefits_harm_society",
        "welfare_chauvinism": "welfare_chauvinism",
        "anti_interventionism": "anti_economic_interventionism",
        "harsh_sentences": "harsh_sentences",
        "anti_mil_democracy": "anti_militant_democracy",
        "science_environment": "no_science_environment_solution",
        "government_spending": "anti_government_spending",
        "regressive_taxes": "regressive_taxes",
        "regressive_benefits": "regressive_benefits",
        "age_prejudice": "age_prejudice",
        "authoritarianism": "authoritarianism",
        "anti_libertarianism": "anti_libertarianism",
    }

    required_reference_columns = {
        "essid",
        "country",
        "education",
        "hhincome",
        "female",
        "age",
        "religious",
        "urbanization",
        "ethnic_minority",
        *r_to_python_belief.keys(),
    }
    missing_reference_columns = sorted(
        required_reference_columns - set(reference_raw.columns)
    )
    if missing_reference_columns:
        raise KeyError(
            "Required columns are missing from df_ESS4.RData: "
            f"{missing_reference_columns}"
        )

    country_text = (
        reference_raw["country"].astype("string").str.strip()
    )
    country_name_to_code = {
        name: code
        for code, name in ess4.COUNTRY_LABELS.items()
    }
    country_name_to_code.update(
        {
            "Great Britain": "GB",
            "United Kingdom": "GB",
            "Czech Republic": "CZ",
            "Czechia": "CZ",
            "Russia": "RU",
            "Russian Federation": "RU",
            "Turkey": "TR",
            "Türkiye": "TR",
        }
    )

    reference = pd.DataFrame(index=reference_raw.index)
    reference["idno"] = pd.to_numeric(
        reference_raw["essid"],
        errors="coerce",
    ).astype("Int64")
    reference["cntry"] = country_text.where(
        country_text.isin(ess4.COUNTRY_LABELS),
        country_text.map(country_name_to_code),
    )

    reference["education_3cat"] = map_factor_to_numeric(
        reference_raw["education"],
        label_mapping={
            "Lower educated": 1,
            "Middle educated": 2,
            "Higher educated": 3,
        },
        numeric_code_mapping={1: 1, 2: 2, 3: 3},
    )
    reference["hinctnta"] = pd.to_numeric(
        reference_raw["hhincome"],
        errors="coerce",
    )
    reference["gndr"] = map_factor_to_numeric(
        reference_raw["female"],
        label_mapping={"Male": 1, "Female": 2},
        numeric_code_mapping={1: 1, 2: 2},
    )
    reference["agea"] = pd.to_numeric(
        reference_raw["age"],
        errors="coerce",
    )
    reference["rlgblg"] = map_factor_to_numeric(
        reference_raw["religious"],
        label_mapping={"Non-religious": 2, "Religious": 1},
        numeric_code_mapping={1: 2, 2: 1},
    )
    reference["urbanization"] = pd.to_numeric(
        reference_raw["urbanization"],
        errors="coerce",
    )
    reference["blgetmg"] = map_factor_to_numeric(
        reference_raw["ethnic_minority"],
        label_mapping={"Ethnic majority": 2, "Ethnic minority": 1},
        numeric_code_mapping={1: 2, 2: 1},
    )

    for r_column, python_column in r_to_python_belief.items():
        reference[python_column] = pd.to_numeric(
            reference_raw[r_column],
            errors="coerce",
        )

    if reference[["cntry", "idno"]].duplicated().any():
        raise AssertionError(
            "Duplicate respondent keys found in the ESS4 R reference."
        )

    python_data = ess4_without.copy()
    python_data["idno"] = pd.to_numeric(
        python_data["idno"],
        errors="coerce",
    ).astype("Int64")
    python_data["cntry"] = (
        python_data["cntry"].astype("string").str.strip()
    )

    if python_data[["cntry", "idno"]].duplicated().any():
        raise AssertionError(
            "Duplicate respondent keys found in the Python ESS4 output."
        )

    respondent_keys = python_data[["cntry", "idno"]].merge(
        reference[["cntry", "idno"]],
        on=["cntry", "idno"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not respondent_keys["_merge"].eq("both").all():
        key_counts = respondent_keys["_merge"].value_counts().to_dict()
        raise AssertionError(
            "The Python and R ESS4 respondent sets differ: "
            f"{key_counts}"
        )

    comparison = python_data.merge(
        reference,
        on=["cntry", "idno"],
        how="inner",
        suffixes=("_python", "_reference"),
        validate="one_to_one",
    )

    demographic_columns = (
        "education_3cat",
        "hinctnta",
        "gndr",
        "agea",
        "rlgblg",
        "urbanization",
        "blgetmg",
    )
    for column in demographic_columns:
        match_mask = numeric_match_mask(
            comparison[f"{column}_python"],
            comparison[f"{column}_reference"],
        )
        if not match_mask.all():
            raise AssertionError(
                f"ESS4 demographic mismatch for {column}: "
                f"{int((~match_mask).sum())} respondents."
            )

    validation_rows = []
    for belief in ess4.BELIEF_COLUMNS:
        python_values = pd.to_numeric(
            comparison[f"{belief}_python"],
            errors="coerce",
        )
        reference_values = pd.to_numeric(
            comparison[f"{belief}_reference"],
            errors="coerce",
        )

        match_mask = numeric_match_mask(
            python_values,
            reference_values,
        )
        both_present = python_values.notna() & reference_values.notna()
        absolute_differences = (
            python_values.loc[both_present]
            - reference_values.loc[both_present]
        ).abs()
        maximum_absolute_difference = (
            float(absolute_differences.max())
            if not absolute_differences.empty
            else 0.0
        )

        validation_rows.append(
            {
                "belief_variable": belief,
                "N_python": int(python_values.notna().sum()),
                "N_reference": int(reference_values.notna().sum()),
                "mean_python": python_values.mean(),
                "mean_reference": reference_values.mean(),
                "sd_python": python_values.std(ddof=1),
                "sd_reference": reference_values.std(ddof=1),
                "missingness_mismatches": int(
                    (
                        python_values.isna()
                        != reference_values.isna()
                    ).sum()
                ),
                "value_mismatches": int((~match_mask).sum()),
                "max_abs_difference": maximum_absolute_difference,
                "N_matches": bool(
                    python_values.notna().sum()
                    == reference_values.notna().sum()
                ),
                "values_match_within_tolerance": bool(match_mask.all()),
            }
        )

    validation = pd.DataFrame(validation_rows)

    if not validation["N_matches"].all():
        raise AssertionError(
            "One or more ESS4 belief non-missing counts do not match."
        )
    if not validation["values_match_within_tolerance"].all():
        failed = validation.loc[
            ~validation["values_match_within_tolerance"]
        ]
        raise AssertionError(
            "One or more ESS4 beliefs differ from df_ESS4.RData:\n"
            + failed.to_string(index=False)
        )

    print(
        "ESS4 validation passed: all 19 belief variables match "
        "df_ESS4.RData respondent by respondent."
    )
    return validation


def create_cross_round_descriptives(
    ess4_without: pd.DataFrame,
    ess8_without: pd.DataFrame,
) -> pd.DataFrame:
    """Describe the belief concepts shared by ESS4 and ESS8."""
    shared_beliefs = tuple(
        belief
        for belief in ess8.BELIEF_COLUMNS
        if belief in ess4.BELIEF_COLUMNS
    )
    if len(shared_beliefs) != 12:
        raise AssertionError(
            f"Expected 12 shared belief concepts, found "
            f"{len(shared_beliefs)}."
        )

    def describe(
        dataframe: pd.DataFrame,
        round_label: str,
    ) -> pd.DataFrame:
        result = summarise_beliefs(
            dataframe,
            shared_beliefs,
        ).rename(
            columns={
                "N_reproduced": "N_nonmissing",
                "mean_reproduced": "mean",
                "sd_reproduced": "sd",
            }
        )
        result.insert(0, "round", round_label)
        result["N_total"] = len(dataframe)
        result["N_missing"] = (
            result["N_total"] - result["N_nonmissing"]
        )
        result["percent_missing"] = (
            100.0 * result["N_missing"] / result["N_total"]
        )
        return result

    return pd.concat(
        [
            describe(ess4_without, ess4.ROUND_LABEL),
            describe(ess8_without, ess8.ROUND_LABEL),
        ],
        ignore_index=True,
    )


def write_and_verify_dataset(
    dataframe: pd.DataFrame,
    destination: Path,
    expected_columns: Sequence[str],
) -> None:
    """Write one CSV and verify its saved schema and row count."""
    dataframe.to_csv(destination, index=False)
    reloaded = pd.read_csv(destination, low_memory=False)

    if reloaded.shape != dataframe.shape:
        raise AssertionError(
            f"Saved file has unexpected shape: {destination}"
        )

    if list(reloaded.columns) != list(expected_columns):
        raise AssertionError(
            f"Saved file has unexpected column order: {destination}"
        )

    if not reloaded["ess_unique_id"].is_unique:
        raise AssertionError(
            f"Saved file has duplicate respondent IDs: {destination}"
        )


def main() -> None:
    """Build, validate, write, and verify every generated dataset."""
    arguments = parse_arguments()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ess4_without, ess4_with = build_round(ess4)
    ess8_without, ess8_with = build_round(ess8)

    ess8_validation = validate_ess8_against_paper(ess8_without)

    ess4_validation = None
    if arguments.skip_ess4_reference_validation:
        print()
        print(
            "WARNING: ESS4 reference validation was skipped by request."
        )
    else:
        ess4_validation = validate_ess4_against_reference(ess4_without)

    cross_round_descriptives = create_cross_round_descriptives(
        ess4_without,
        ess8_without,
    )

    print()
    print("=" * 72)
    print("Writing processed outputs")
    print("=" * 72)

    write_and_verify_dataset(
        ess4_without,
        output_path(ess4, "without_weights"),
        ess4.OUTPUT_COLUMNS_WITHOUT_WEIGHTS,
    )
    write_and_verify_dataset(
        ess4_with,
        output_path(ess4, "with_weights"),
        ess4.OUTPUT_COLUMNS_WITH_WEIGHTS,
    )
    write_and_verify_dataset(
        ess8_without,
        output_path(ess8, "without_weights"),
        ess8.OUTPUT_COLUMNS_WITHOUT_WEIGHTS,
    )
    write_and_verify_dataset(
        ess8_with,
        output_path(ess8, "with_weights"),
        ess8.OUTPUT_COLUMNS_WITH_WEIGHTS,
    )

    ess8_validation.to_csv(
        output_path(ess8, "validation"),
        index=False,
    )
    if ess4_validation is not None:
        ess4_validation.to_csv(
            output_path(ess4, "validation"),
            index=False,
        )

    cross_round_path = (
        PROCESSED_DIR / CROSS_ROUND_DESCRIPTIVES_FILENAME
    )
    cross_round_descriptives.to_csv(
        cross_round_path,
        index=False,
    )

    validate_weight_split(
        pd.read_csv(
            output_path(ess4, "without_weights"),
            low_memory=False,
        ),
        pd.read_csv(
            output_path(ess4, "with_weights"),
            low_memory=False,
        ),
        ess4.WEIGHT_COLUMNS,
        require_complete_weights=True,
    )
    validate_weight_split(
        pd.read_csv(
            output_path(ess8, "without_weights"),
            low_memory=False,
        ),
        pd.read_csv(
            output_path(ess8, "with_weights"),
            low_memory=False,
        ),
        ess8.WEIGHT_COLUMNS,
        require_complete_weights=True,
    )

    print()
    print("All requested datasets were built successfully.")
    print()
    print("Principal outputs:")
    print("-", output_path(ess4, "without_weights"))
    print("-", output_path(ess4, "with_weights"))
    print("-", output_path(ess8, "without_weights"))
    print("-", output_path(ess8, "with_weights"))
    print()
    print("Validation outputs:")
    if ess4_validation is not None:
        print("-", output_path(ess4, "validation"))
    print("-", output_path(ess8, "validation"))
    print("-", cross_round_path)


if __name__ == "__main__":
    main()
