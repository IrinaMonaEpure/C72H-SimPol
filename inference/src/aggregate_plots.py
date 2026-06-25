from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def extract_country_code(path):
    stem = Path(path).stem
    return stem.split("_")[-1]


def load_weight_matrix(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".pkl", ".pickle"]:
        obj = pd.read_pickle(path)

        if isinstance(obj, pd.DataFrame):
            return obj

        if isinstance(obj, np.ndarray):
            return pd.DataFrame(obj)

        raise ValueError(f"Unsupported pickle contents in {path}")

    if suffix == ".csv":
        return pd.read_csv(path, index_col=0)

    if suffix == ".graphml":
        G = nx.read_graphml(path)
        return nx.to_pandas_adjacency(G, weight="weight", dtype=float)

    raise ValueError(
        f"Unsupported file format: {path}. Use .pkl, .pickle, .csv, or .graphml."
    )


def get_off_diagonal_weights(matrix):
    A = matrix.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    values = A[mask]
    return values[~np.isnan(values)]


def plot_all_correlation_distributions(
    folder,
    output_path,
    pattern="*",
    bins=40,
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    title="Correlation weight distributions",
    show=False,
):
    folder = Path(folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise ValueError(f"No supported weight files found in {folder}")

    loaded = []

    for path in paths:
        matrix = load_weight_matrix(path)
        values = get_off_diagonal_weights(matrix)

        if len(values) == 0:
            continue

        country_code = extract_country_code(path)

        loaded.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "path": path,
                "values": values,
            }
        )

    if not loaded:
        raise ValueError("No valid weights found.")

    loaded = sorted(loaded, key=lambda x: x["country_code"])

    all_values = np.concatenate([item["values"] for item in loaded])

    x_min = float(np.nanmin(all_values))
    x_max = float(np.nanmax(all_values))

    pad = 0.05 * max(abs(x_max - x_min), 1e-6)
    xlim = (x_min - pad, x_max + pad)

    ymax = 0

    for item in loaded:
        counts, _ = np.histogram(
            item["values"],
            bins=bins,
            range=xlim,
        )
        ymax = max(ymax, counts.max())

    ylim = (0, ymax * 1.1)

    n = len(loaded)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, max(4, 3.2 * nrows)),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, loaded):
        sns.histplot(
            item["values"],
            bins=bins,
            binrange=xlim,
            kde=True,
            ax=ax,
        )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(item["country_code"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")

    for ax in axes[len(loaded):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return loaded


def plot_first_column_sums(
    folder,
    output_plot_path,
    output_table_path,
    output_first_columns_plot_path=None,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    use_absolute=False,
    title="First-column weight sums by country",
    first_columns_title="First-column weights by country",
    show=False,
):
    folder = Path(folder)
    output_plot_path = Path(output_plot_path)
    output_table_path = Path(output_table_path)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)

    if output_first_columns_plot_path is None:
        output_first_columns_plot_path = (
            output_plot_path.parent /
            f"{output_plot_path.stem}_first_columns.png"
        )
    else:
        output_first_columns_plot_path = Path(output_first_columns_plot_path)

    output_first_columns_plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise ValueError(f"No supported matrix files found in {folder}")

    rows = []
    first_column_rows = []

    for path in paths:
        matrix = load_weight_matrix(path)

        if matrix.shape[1] == 0:
            continue

        country_code = extract_country_code(path)

        first_col = matrix.columns[0]
        values = pd.to_numeric(matrix[first_col], errors="coerce")

        if first_col in values.index:
            values_no_self = values.drop(index=first_col)
        else:
            values_no_self = values

        if use_absolute:
            col_sum = values_no_self.abs().sum(skipna=True)
        else:
            col_sum = values_no_self.sum(skipna=True)

        rows.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "first_column": first_col,
                "column_sum": col_sum,
            }
        )

        for target, weight in values_no_self.items():
            first_column_rows.append(
                {
                    "country_code": country_code,
                    "weights_name": path.stem,
                    "first_column": first_col,
                    "target": target,
                    "weight": weight,
                }
            )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "column_sum",
        ascending=False,
    ).reset_index(drop=True)

    ordered_countries = result["country_code"].tolist()

    first_columns = pd.DataFrame(first_column_rows)
    first_columns["country_code"] = pd.Categorical(
        first_columns["country_code"],
        categories=ordered_countries,
        ordered=True,
    )

    first_columns = first_columns.sort_values(
        ["country_code", "target"]
    )

    suffix = output_table_path.suffix.lower()

    if suffix == ".csv":
        result.to_csv(output_table_path, index=False)
    elif suffix in [".pkl", ".pickle"]:
        result.to_pickle(output_table_path)
    else:
        raise ValueError("Use .csv or .pkl for output_table_path")

    # Plot 1: column sums, countries on x-axis
    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result)), 6),
    )

    sns.barplot(
        data=result,
        x="country_code",
        y="column_sum",
        ax=ax,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Country code")
    ax.set_ylabel(
        "Sum of first-column weights"
        if not use_absolute
        else "Sum of absolute first-column weights"
    )

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    fig.savefig(
        output_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Plot 2: all first-column weights ordered by country-sum
    fig, ax = plt.subplots(
        figsize=(max(12, 0.5 * len(ordered_countries)), 7),
    )

    sns.lineplot(
        data=first_columns,
        x="country_code",
        y="weight",
        hue="target",
        marker="o",
        linewidth=1.2,
        alpha=0.85,
        ax=ax,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(first_columns_title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Country code, ordered by first-column sum")
    ax.set_ylabel("First-column weight")

    ax.tick_params(axis="x", rotation=45)

    # Move legend outside; useful if many target columns.
    ax.legend(
        title="Target",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    fig.savefig(
        output_first_columns_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return result