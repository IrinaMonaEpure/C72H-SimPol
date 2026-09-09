from pathlib import Path

import pandas as pd


def aggregate_polarization(
    input_csv="../../../abm/output/polarization_mdl.csv",
    output_csv="../../../abm/output/polarization_mdl_aggregated.csv",
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    required_columns = {
        "country",
        "beta_internal",
        "beta_external",
        "polarization",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    aggregated = (
        df.groupby(
            [
                "country",
                "beta_internal",
                "beta_external",
            ],
            as_index=False,
        )
        .agg(
            polarization_mean=("polarization", "mean"),
            polarization_std=("polarization", "std"),
        )
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated.to_csv(
        output_csv,
        index=False,
    )

    print(f"Saved aggregated results to: {output_csv}")

    return aggregated


if __name__ == "__main__":
    aggregate_polarization()