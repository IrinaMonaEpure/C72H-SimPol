from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram


def extract_country_code(path):
    stem = Path(path).stem
    return stem.split("_")[-1]


def correlation_value_to_hex(value, cmap="RdBu", vmin=-1, vmax=1):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = mpl.colormaps[cmap]
    rgba = colormap(norm(value))
    return mpl.colors.to_hex(rgba)


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

        node_labels = {
            node: data.get("belief", node)
            for node, data in G.nodes(data=True)
        }

        matrix = nx.to_pandas_adjacency(
            G,
            weight="weight",
            dtype=float,
        )

        return matrix.rename(
            index=node_labels,
            columns=node_labels,
        )

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

        loaded.append(
            {
                "country_code": extract_country_code(path),
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
        figsize=(14, max(4, 2.5 * nrows)),
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
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[len(loaded):]:
        ax.axis("off")

    fig.supxlabel("Correlation weight", fontsize=12)
    fig.supylabel("Count", fontsize=12)

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


def plot_first_column_averages(
    folder,
    output_plot_path,
    output_table_path,
    output_first_columns_plot_path=None,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    use_absolute=False,
    title="Average first-column weight by country",
    first_columns_title="First-column weights by country",
    cmap="RdBu",
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
            column_average = values_no_self.abs().mean(skipna=True)
            color_value = column_average
        else:
            column_average = values_no_self.mean(skipna=True)
            color_value = column_average

        average_color_hex = correlation_value_to_hex(
            color_value,
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )

        rows.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "first_column": first_col,
                "column_average": column_average,
                "average_color_hex": average_color_hex,
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
        "column_average",
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

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result)), 6),
    )

    ax.bar(
        result["country_code"],
        result["column_average"],
        color=result["average_color_hex"].tolist(),
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Country code")
    ax.set_ylabel(
        "Average first-column weight"
        if not use_absolute
        else "Average absolute first-column weight"
    )
    ax.tick_params(axis="x", rotation=45)

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = mpl.cm.ScalarMappable(
        cmap=mpl.colormaps[cmap],
        norm=norm,
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Correlation scale")

    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    first_column_matrix = first_columns.pivot(
        index="target",
        columns="country_code",
        values="weight",
    )

    first_column_matrix = first_column_matrix[ordered_countries]

    fig, ax = plt.subplots(
        figsize=(
            max(12, 0.5 * len(ordered_countries)),
            max(6, 0.35 * len(first_column_matrix)),
        ),
    )

    sns.heatmap(
        first_column_matrix,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Correlation with left_right_identification"},
        ax=ax,
    )

    ax.set_xlabel("Country code")
    ax.set_ylabel("Target variable")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

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


def plot_total_correlation_summaries(
    folder,
    output_signed_plot_path,
    output_absolute_plot_path,
    output_table_path,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    bar_color="#4C72B0",
    show=False,
):
    folder = Path(folder)

    output_signed_plot_path = Path(output_signed_plot_path)
    output_absolute_plot_path = Path(output_absolute_plot_path)
    output_table_path = Path(output_table_path)

    output_signed_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_absolute_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise ValueError(f"No supported matrix files found in {folder}")

    rows = []

    for path in paths:
        matrix = load_weight_matrix(path)
        values = get_off_diagonal_weights(matrix)

        if len(values) == 0:
            continue

        country_code = extract_country_code(path)

        rows.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "n_edges": len(values),
                "signed_correlation_sum": np.nansum(values),
                "absolute_correlation_sum": np.nansum(np.abs(values)),
                "mean_correlation": np.nanmean(values),
                "mean_absolute_correlation": np.nanmean(np.abs(values)),
                "min_correlation": np.nanmin(values),
                "max_correlation": np.nanmax(values),
                "std_correlation": np.nanstd(values),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No valid matrices found.")

    result_signed = result.sort_values(
        "signed_correlation_sum",
        ascending=False,
    ).reset_index(drop=True)

    result_absolute = result.sort_values(
        "absolute_correlation_sum",
        ascending=False,
    ).reset_index(drop=True)

    suffix = output_table_path.suffix.lower()

    if suffix == ".csv":
        result.to_csv(output_table_path, index=False)
    elif suffix in [".pkl", ".pickle"]:
        result.to_pickle(output_table_path)
    else:
        raise ValueError("output_table_path must be .csv or .pkl")

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result_signed)), 6)
    )

    ax.bar(
        result_signed["country_code"],
        result_signed["signed_correlation_sum"],
        color=bar_color,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Country code")
    ax.set_ylabel("Sum of correlation values")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_signed_plot_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result_absolute)), 6)
    )

    ax.bar(
        result_absolute["country_code"],
        result_absolute["absolute_correlation_sum"],
        color=bar_color,
    )

    ax.set_xlabel("Country code")
    ax.set_ylabel("Sum of absolute correlation values")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_absolute_plot_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return result


def save_top_beliefs_by_absolute_edge_sum(
    folder,
    output_csv_path,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    top_n=5,
    ignore_belief=None,
    return_dataframe=False,
):
    folder = Path(folder)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise ValueError(
            f"No supported matrix files found in {folder}"
        )

    rows = []

    for path in paths:
        matrix = load_weight_matrix(path)

        if matrix.empty:
            continue

        country_code = extract_country_code(path)

        A = matrix.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Remove the selected belief entirely.
        if ignore_belief is not None:

            if ignore_belief in A.index:
                A = A.drop(
                    index=ignore_belief,
                    columns=ignore_belief,
                )

        if A.empty:
            continue

        # Make a writable copy.
        A_values = A.to_numpy(
            dtype=float,
            copy=True,
        )

        # Remove self-connections.
        np.fill_diagonal(
            A_values,
            0,
        )

        belief_strengths = pd.Series(
            np.abs(A_values).sum(axis=1),
            index=A.index,
        )

        top_beliefs = belief_strengths.sort_values(
            ascending=False,
        ).head(top_n)

        for rank, (belief, absolute_edge_sum) in enumerate(
            top_beliefs.items(),
            start=1,
        ):
            rows.append(
                {
                    "country_code": country_code,
                    "weights_name": path.stem,
                    "rank": rank,
                    "belief": belief,
                    "absolute_edge_sum": absolute_edge_sum,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No valid belief edge sums found."
        )

    result.to_csv(
        output_csv_path,
        index=False,
    )

    if return_dataframe:
        return result

    return None


def plot_top_beliefs_stacked_by_country(
    top_beliefs_csv_path,
    ordering_csv_path,
    output_path,
    country_col="country_code",
    belief_col="belief",
    rank_col="rank",
    ordering_col="column_average",
    show=False,
):
    top_beliefs = pd.read_csv(top_beliefs_csv_path)
    ordering = pd.read_csv(ordering_csv_path)

    ordered_countries = (
        ordering
        .sort_values(ordering_col, ascending=False)[country_col]
        .tolist()
    )

    top_beliefs = top_beliefs[
        top_beliefs[country_col].isin(ordered_countries)
    ].copy()

    top_beliefs[country_col] = pd.Categorical(
        top_beliefs[country_col],
        categories=ordered_countries,
        ordered=True,
    )

    top_beliefs = top_beliefs.sort_values(
        [country_col, rank_col]
    )

    countries = ordered_countries
    beliefs = sorted(top_beliefs[belief_col].dropna().unique())

    cmap = plt.get_cmap("tab20")
    belief_colors = {
        belief: cmap(i % cmap.N)
        for i, belief in enumerate(beliefs)
    }

    fig, ax = plt.subplots(
        figsize=(max(12, 0.5 * len(countries)), 7)
    )

    bottom = np.zeros(len(countries))

    for rank in sorted(top_beliefs[rank_col].unique()):
        rank_data = top_beliefs[top_beliefs[rank_col] == rank]

        heights = np.zeros(len(countries))
        colors = []

        for i, country in enumerate(countries):
            row = rank_data[rank_data[country_col] == country]

            if row.empty:
                heights[i] = 0
                colors.append("white")
            else:
                belief = row.iloc[0][belief_col]
                heights[i] = 1
                colors.append(belief_colors[belief])

        ax.bar(
            countries,
            heights,
            bottom=bottom,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )

        bottom += heights

    ax.set_xlabel("Country code")
    ax.set_ylabel("Top 5 belief rank slots")
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(axis="x", rotation=45)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for belief, color in belief_colors.items()
    ]

    ax.legend(
        handles,
        beliefs,
        title="Belief",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def plot_near_zero_correlation_counts(
    folder,
    output_plot_path,
    output_csv_path,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    zero_threshold=0.05,
    bar_color="#4C72B0",
    show=False,
    return_dataframe=False,
):
    """
    Count near-zero correlations per country and plot them.

    A correlation is considered near zero when:

        abs(correlation) <= zero_threshold

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder containing network matrices.
    output_plot_path : str or pathlib.Path
        Where to save the bar plot.
    output_csv_path : str or pathlib.Path
        Where to save the summary CSV.
    pattern : str
        Glob pattern for files.
    extensions : tuple
        Supported matrix file extensions.
    zero_threshold : float
        Absolute-value threshold for counting near-zero correlations.
        Example: 0.05 counts values between -0.05 and 0.05.
    bar_color : str
        Bar color.
    show : bool
        Whether to show the plot.
    return_dataframe : bool
        Whether to return the summary dataframe.

    Returns
    -------
    pandas.DataFrame or None
    """

    folder = Path(folder)
    output_plot_path = Path(output_plot_path)
    output_csv_path = Path(output_csv_path)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise ValueError(f"No supported matrix files found in {folder}")

    rows = []

    for path in paths:
        matrix = load_weight_matrix(path)
        values = get_off_diagonal_weights(matrix)

        if len(values) == 0:
            continue

        country_code = extract_country_code(path)

        near_zero_mask = np.abs(values) <= zero_threshold

        near_zero_count = int(np.sum(near_zero_mask))
        total_edges = int(len(values))

        rows.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "zero_threshold": zero_threshold,
                "near_zero_count": near_zero_count,
                "total_edges": total_edges,
                "near_zero_fraction": near_zero_count / total_edges,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No valid correlation values found.")

    result = result.sort_values(
        "near_zero_count",
        ascending=False,
    ).reset_index(drop=True)

    result.to_csv(output_csv_path, index=False)

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result)), 6)
    )

    ax.bar(
        result["country_code"],
        result["near_zero_count"],
        color=bar_color,
    )

    ax.set_xlabel("Country code")
    ax.set_ylabel(
        f"Number of correlations with |r| ≤ {zero_threshold}"
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

    if return_dataframe:
        return result

    return None


def _build_linkage_from_dendrogram_csv(dendrogram_csv, leaf_labels):
    dendro_df = pd.read_csv(dendrogram_csv)
    dendro_df = dendro_df.sort_values("merge_step")

    leaf_to_id = {
        label: i
        for i, label in enumerate(leaf_labels)
    }

    cluster_to_linkage_id = {}
    next_linkage_id = len(leaf_labels)

    linkage_rows = []

    for _, row in dendro_df.iterrows():
        child_linkage_ids = []

        for child_id_col, child_label_col in [
            ("child_1_id", "child_1_label"),
            ("child_2_id", "child_2_label"),
        ]:
            child_id = row[child_id_col]
            child_label = row[child_label_col]

            if child_label in leaf_to_id:
                child_linkage_ids.append(leaf_to_id[child_label])
            elif child_id in cluster_to_linkage_id:
                child_linkage_ids.append(cluster_to_linkage_id[child_id])
            elif str(child_id) in cluster_to_linkage_id:
                child_linkage_ids.append(cluster_to_linkage_id[str(child_id)])
            elif child_label in cluster_to_linkage_id:
                child_linkage_ids.append(cluster_to_linkage_id[child_label])
            elif str(child_label) in cluster_to_linkage_id:
                child_linkage_ids.append(cluster_to_linkage_id[str(child_label)])
            else:
                raise ValueError(
                    f"Could not resolve dendrogram child: "
                    f"{child_id_col}={child_id}, "
                    f"{child_label_col}={child_label}"
                )

        linkage_rows.append(
            [
                child_linkage_ids[0],
                child_linkage_ids[1],
                float(row["distance"]),
                int(row["n_members"]),
            ]
        )

        cluster_to_linkage_id[row["cluster_id"]] = next_linkage_id
        cluster_to_linkage_id[str(row["cluster_id"])] = next_linkage_id

        next_linkage_id += 1

    return np.asarray(linkage_rows, dtype=float)


def plot_first_column_average_with_dendrogram(
    folder,
    dendrogram_csv,
    output_plot_path,
    output_table_path,
    output_first_columns_plot_path=None,
    pattern="*",
    extensions=(".pkl", ".pickle", ".csv", ".graphml"),
    use_absolute=False,
    title="Average first-column weight by country",
    first_columns_title="First-column weights by country",
    cmap="RdBu",
    show=False,
    dendrogram_width=1.0,
):
    folder = Path(folder)
    dendrogram_csv = Path(dendrogram_csv)
    output_plot_path = Path(output_plot_path)
    output_table_path = Path(output_table_path)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)

    if output_first_columns_plot_path is None:
        output_first_columns_plot_path = (
            output_plot_path.parent
            / f"{output_plot_path.stem}_first_columns_with_dendrogram.png"
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
            column_average = values_no_self.abs().mean(skipna=True)
            color_value = column_average
        else:
            column_average = values_no_self.mean(skipna=True)
            color_value = column_average

        average_color_hex = correlation_value_to_hex(
            color_value,
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )

        rows.append(
            {
                "country_code": country_code,
                "weights_name": path.stem,
                "first_column": first_col,
                "column_average": column_average,
                "average_color_hex": average_color_hex,
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

    if result.empty:
        raise ValueError("No valid first-column values found.")

    result = result.sort_values(
        "column_average",
        ascending=False,
    ).reset_index(drop=True)

    original_ordered_countries = result["country_code"].tolist()

    first_columns = pd.DataFrame(first_column_rows)

    suffix = output_table_path.suffix.lower()

    if suffix == ".csv":
        result.to_csv(output_table_path, index=False)
    elif suffix in [".pkl", ".pickle"]:
        result.to_pickle(output_table_path)
    else:
        raise ValueError("Use .csv or .pkl for output_table_path")

    # --------------------------------------------------
    # Plot 1: average first-column bar plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(result)), 6),
    )

    ax.bar(
        result["country_code"],
        result["column_average"],
        color=result["average_color_hex"].tolist(),
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Country code")
    ax.set_ylabel(
        "Average first-column weight"
        if not use_absolute
        else "Average absolute first-column weight"
    )
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --------------------------------------------------
    # Plot 2: first-column matrix + clustering dendrogram
    # --------------------------------------------------

    first_column_matrix = first_columns.pivot(
        index="target",
        columns="country_code",
        values="weight",
    )

    Z = _build_linkage_from_dendrogram_csv(
        dendrogram_csv=dendrogram_csv,
        leaf_labels=original_ordered_countries,
    )

    ddata = dendrogram(
        Z,
        labels=original_ordered_countries,
        orientation="bottom",
        no_plot=True,
    )

    clustering_ordered_countries = ddata["ivl"]

    first_column_matrix = first_column_matrix[
        clustering_ordered_countries
    ]

    fig = plt.figure(
        figsize=(
            max(12, 0.5 * len(clustering_ordered_countries)),
            max(8, 0.35 * len(first_column_matrix) + 3),
        )
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[5, 1.7],
        hspace=0.1,
    )

    ax_heatmap = fig.add_subplot(gs[0])

    vmin = np.nanmin(first_column_matrix.values)
    vmax = np.nanmax(first_column_matrix.values)

    sns.heatmap(
        first_column_matrix,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={
            "label": "Correlation with left_right_identification"
        },
        ax=ax_heatmap,
    )

    #ax_heatmap.set_title(first_columns_title)
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("Target variable")
    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )
    ax_heatmap.tick_params(axis="y", rotation=0)

    ax_dendro = fig.add_subplot(gs[1])

    dendrogram(
        Z,
        labels=original_ordered_countries,
        orientation="bottom",
        ax=ax_dendro,
        leaf_rotation=45,
        color_threshold=0,
        above_threshold_color="black",
    )

    ax_dendro.set_xlabel("Country code")
    ax_dendro.set_ylabel("Distance")

    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)
    ax_dendro.spines["left"].set_visible(False)
    ax_dendro.spines["bottom"].set_visible(False)

    ax_dendro.tick_params(axis="x", length=0)
    ax_dendro.tick_params(axis="y", length=0)

    fig.tight_layout()

    fig.canvas.draw()

    heatmap_pos = ax_heatmap.get_position()
    dendro_pos = ax_dendro.get_position()

    dendrogram_width = 1 # smaller = narrower

    new_width = heatmap_pos.width * dendrogram_width
    new_x0 = heatmap_pos.x0

    ax_dendro.set_position([
        new_x0,
        dendro_pos.y0,
        new_width,
        dendro_pos.height,
    ])

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