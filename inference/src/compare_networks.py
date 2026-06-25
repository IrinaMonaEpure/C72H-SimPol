"""Compare belief networks across countries.

Computes pairwise network distances, balance indices, and signed community
structure, then produces clustering, MDS, and statistical test outputs to
answer whether Eastern/Western European countries share similar belief
network structure.

Usage (from inference/src/):
    python compare_networks.py

Reads : ../output/networks/<cntry>.graphml
Writes: ../output/comparison/
"""

import glob
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.linalg import expm
from sklearn.manifold import MDS

NETWORK_DIR = Path(__file__).resolve().parent.parent / "output" / "networks"
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "comparison"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}

EASTERN = {"CZ", "EE", "HU", "LT", "PL", "RU", "SI"}
WESTERN = {"AT", "BE", "CH", "DE", "ES", "FI", "FR", "GB", "IE",
           "IS", "IT", "NL", "NO", "PT", "SE"}

REGION_LABELS = {c: "Eastern" for c in EASTERN}
REGION_LABELS.update({c: "Western" for c in WESTERN})
REGION_LABELS["IL"] = "Other"

REGION_COLORS = {"Eastern": "#d62728", "Western": "#1f77b4", "Other": "#7f7f7f"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_network(path):
    """Load a GraphML file and return (adjacency_matrix, belief_labels)."""
    G = nx.read_graphml(path)
    beliefs = [G.nodes[n].get("belief", n) for n in sorted(G.nodes(),
               key=lambda x: int(x[1:]))]
    n = len(beliefs)
    matrix = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = int(u[1:]), int(v[1:])
        w = float(d.get("weight", 0))
        matrix[i, j] = w
        matrix[j, i] = w
    return matrix, beliefs


def load_all_networks():
    """Load all country networks, return dict of {code: matrix} and labels."""
    networks = {}
    beliefs = None
    for path in sorted(glob.glob(str(NETWORK_DIR / "*.graphml"))):
        code = Path(path).stem
        mat, b = load_network(path)
        networks[code] = mat
        if beliefs is None:
            beliefs = b
    return networks, beliefs


# ---------------------------------------------------------------------------
# Layer 1: Edge-weight pairwise distances
# ---------------------------------------------------------------------------

def upper_triangle(mat):
    """Extract upper triangle as a flat vector (excludes diagonal)."""
    return mat[np.triu_indices_from(mat, k=1)]


def frobenius_distance_matrix(networks):
    """Compute pairwise Frobenius distance between partial-correlation matrices."""
    codes = list(networks.keys())
    n = len(codes)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(networks[codes[i]] - networks[codes[j]])
        dist[i, j] = d
        dist[j, i] = d
    return pd.DataFrame(dist, index=codes, columns=codes)


def cosine_similarity_matrix(networks):
    """Compute pairwise cosine similarity of flattened upper-triangle vectors."""
    codes = list(networks.keys())
    vecs = {c: upper_triangle(m) for c, m in networks.items()}
    n = len(codes)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = vecs[codes[i]], vecs[codes[j]]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            sim[i, j] = np.dot(a, b) / denom if denom > 0 else 0
    return pd.DataFrame(sim, index=codes, columns=codes)


def correlation_similarity_matrix(networks):
    """Pairwise Spearman correlation of upper-triangle edge-weight vectors."""
    codes = list(networks.keys())
    vecs = {c: upper_triangle(m) for c, m in networks.items()}
    n = len(codes)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            else:
                rho, _ = spearmanr(vecs[codes[i]], vecs[codes[j]])
                sim[i, j] = rho
    return pd.DataFrame(sim, index=codes, columns=codes)


# ---------------------------------------------------------------------------
# Layer 2: Signed network structural properties
# ---------------------------------------------------------------------------

def sign_adjacency(mat):
    """Convert weighted matrix to signed adjacency (+1, -1, 0)."""
    return np.sign(mat)


def triangle_balance_index(mat):
    """T = tr(A^3) / tr(|A|^3), from Diaz-Diaz et al. Sec 3.5."""
    A = sign_adjacency(mat)
    A3 = A @ A @ A
    absA = np.abs(A)
    absA3 = absA @ absA @ absA
    denom = np.trace(absA3)
    return np.trace(A3) / denom if denom != 0 else 0


def walk_balance_index(mat, beta=1.0):
    """Estrada-Benzi index: kappa = tr(exp(beta*A)) / tr(exp(beta*|A|)).

    Bounded [0, 1]. 1 = perfectly balanced.
    """
    A = sign_adjacency(mat)
    absA = np.abs(A)
    num = np.trace(expm(beta * A))
    denom = np.trace(expm(beta * absA))
    return num / denom if denom != 0 else 0


def signed_clustering_coefficient(mat):
    """C = 6(t+ - t-) / sum_i k_i(k_i - 1), from Diaz-Diaz Sec 2.2.2."""
    A = sign_adjacency(mat)
    A3 = A @ A @ A
    absA = np.abs(A)
    degrees = absA.sum(axis=0)
    denom = np.sum(degrees * (degrees - 1))
    if denom == 0:
        return 0
    t_plus_minus_t_neg = np.trace(A3)
    return 6 * t_plus_minus_t_neg / denom


def opposing_laplacian_balance(mat):
    """Smallest eigenvalue of the opposing Laplacian L_o.

    L_o: diagonal = sum(|A_ij|), off-diagonal = -A_ij.
    mu_1 = 0 iff network is balanced (Diaz-Diaz Sec 2.3).
    """
    A = sign_adjacency(mat)
    D = np.diag(np.abs(A).sum(axis=1))
    L_o = D - A
    eigvals = np.linalg.eigvalsh(L_o)
    return eigvals[0]


def positive_negative_edge_ratio(mat):
    """Fraction of edges that are positive."""
    edges = mat[np.triu_indices_from(mat, k=1)]
    nonzero = edges[edges != 0]
    if len(nonzero) == 0:
        return 0
    return (nonzero > 0).sum() / len(nonzero)


def network_density(mat):
    """Fraction of possible edges that are present."""
    n = mat.shape[0]
    max_edges = n * (n - 1) / 2
    present = np.count_nonzero(mat[np.triu_indices_from(mat, k=1)])
    return present / max_edges


def signed_triad_census(mat):
    """Count the 4 signed triad types: +++, ++-, +--, ---.

    Returns counts normalized to fractions.
    """
    A = sign_adjacency(mat)
    n = A.shape[0]
    counts = {"+++": 0, "++-": 0, "+--": 0, "---": 0}

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                edges = [A[i, j], A[j, k], A[i, k]]
                nonzero = [e for e in edges if e != 0]
                if len(nonzero) < 3:
                    continue
                n_pos = sum(1 for e in nonzero if e > 0)
                n_neg = 3 - n_pos
                key = "+" * n_pos + "-" * n_neg
                counts[key] += 1

    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def compute_all_properties(networks):
    """Compute structural properties for all countries."""
    rows = []
    for code, mat in networks.items():
        triads = signed_triad_census(mat)
        rows.append({
            "country": code,
            "name": COUNTRY_NAMES.get(code, code),
            "region": REGION_LABELS.get(code, "Other"),
            "n_edges": np.count_nonzero(mat[np.triu_indices_from(mat, k=1)]),
            "density": network_density(mat),
            "pos_edge_ratio": positive_negative_edge_ratio(mat),
            "triangle_balance": triangle_balance_index(mat),
            "walk_balance": walk_balance_index(mat),
            "signed_clustering": signed_clustering_coefficient(mat),
            "laplacian_balance": opposing_laplacian_balance(mat),
            "triad_+++": triads["+++"],
            "triad_++-": triads["++-"],
            "triad_+--": triads["+--"],
            "triad_---": triads["---"],
        })
    return pd.DataFrame(rows).set_index("country")


# ---------------------------------------------------------------------------
# Layer 3: Community structure comparison
# ---------------------------------------------------------------------------

def spectral_partition(mat):
    """Bipartition beliefs using the opposing Laplacian's smallest eigenvector.

    Beliefs are assigned to two factions based on the sign of the Fiedler-like
    eigenvector of L_o (Diaz-Diaz Sec 4.4.2).
    """
    A = sign_adjacency(mat)
    D = np.diag(np.abs(A).sum(axis=1))
    L_o = D - A
    eigvals, eigvecs = np.linalg.eigh(L_o)
    fiedler = eigvecs[:, 0]
    return (fiedler >= 0).astype(int)


def partition_similarity(p1, p2):
    """Normalized Mutual Information between two partitions.

    For binary partitions, accounts for label permutation.
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(p1, p2)


def community_similarity_matrix(networks):
    """Compute pairwise NMI between spectral partitions."""
    codes = list(networks.keys())
    partitions = {c: spectral_partition(m) for c, m in networks.items()}
    n = len(codes)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = partition_similarity(
                partitions[codes[i]], partitions[codes[j]])
    return pd.DataFrame(sim, index=codes, columns=codes)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mantel_test(dist_matrix, group_labels, n_perm=10000):
    """Permutation test: are within-group distances smaller than between-group?

    Returns observed difference (mean_between - mean_within), p-value.
    """
    codes = list(dist_matrix.index)
    D = dist_matrix.values
    n = len(codes)

    within_mask = np.zeros((n, n), dtype=bool)
    between_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if group_labels[codes[i]] == group_labels[codes[j]]:
                within_mask[i, j] = True
            else:
                between_mask[i, j] = True

    within_vals = D[within_mask]
    between_vals = D[between_mask]

    if len(within_vals) == 0 or len(between_vals) == 0:
        return 0, 1.0

    observed = between_vals.mean() - within_vals.mean()

    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(codes)
        perm_labels = {perm[i]: group_labels[codes[i]] for i in range(n)}
        w, b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if perm_labels[codes[i]] == perm_labels[codes[j]]:
                    w.append(D[i, j])
                else:
                    b.append(D[i, j])
        if len(w) > 0 and len(b) > 0:
            perm_diff = np.mean(b) - np.mean(w)
            if perm_diff >= observed:
                count += 1

    return observed, (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distance_heatmap(dist_df, title, path, cmap="viridis_r"):
    """Heatmap of pairwise distances with countries labeled by region."""
    order = sorted(dist_df.index,
                   key=lambda c: (REGION_LABELS.get(c, "Z"), c))
    dist_ordered = dist_df.loc[order, order]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(dist_ordered, cmap=cmap, square=True,
                linewidths=0.3, linecolor="white", ax=ax)

    labels = [f"{c} ({COUNTRY_NAMES.get(c, c)})" for c in order]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)

    colors = [REGION_COLORS.get(REGION_LABELS.get(c, "Other"), "gray")
              for c in order]
    for i, (tick, color) in enumerate(zip(ax.get_yticklabels(), colors)):
        tick.set_color(color)
    for i, (tick, color) in enumerate(zip(ax.get_xticklabels(), colors)):
        tick.set_color(color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(sim_df, title, path):
    """Heatmap for similarity matrices (cosine, correlation, NMI)."""
    plot_distance_heatmap(sim_df, title, path, cmap="RdBu")


def plot_dendrogram(dist_df, title, path, method="ward"):
    """Hierarchical clustering dendrogram colored by East/West.

    Returns (Z, codes) — the linkage matrix and country order.
    """
    codes = list(dist_df.index)
    condensed = squareform(dist_df.values)
    Z = linkage(condensed, method=method)

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [f"{c} ({COUNTRY_NAMES.get(c, c)})" for c in codes]
    dn = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45,
                    leaf_font_size=9)

    leaf_order = dn["ivl"]
    for lbl in ax.get_xticklabels():
        code = lbl.get_text().split(" ")[0]
        region = REGION_LABELS.get(code, "Other")
        lbl.set_color(REGION_COLORS.get(region, "gray"))
        lbl.set_fontweight("bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Distance")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=REGION_COLORS["Eastern"],
               label="Eastern Europe", markersize=10),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=REGION_COLORS["Western"],
               label="Western Europe", markersize=10),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=REGION_COLORS["Other"],
               label="Other", markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return Z, codes


def save_dendrogram_csv(Z, codes, path):
    """Save the linkage matrix as a readable CSV.

    Each row represents a merge step. Cluster IDs 0..n-1 are the original
    countries (mapped to codes); IDs >= n are intermediate clusters formed
    by earlier merges.
    """
    n = len(codes)
    rows = []
    for i, (c1, c2, dist, size) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        label1 = codes[c1] if c1 < n else f"cluster_{c1}"
        label2 = codes[c2] if c2 < n else f"cluster_{c2}"
        rows.append({
            "merge_step": i + 1,
            "cluster_id": n + i,
            "child_1_id": c1,
            "child_1_label": label1,
            "child_2_id": c2,
            "child_2_label": label2,
            "distance": dist,
            "n_members": int(size),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_mds(dist_df, title, path):
    """2D MDS embedding of countries based on network distance."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto")
    coords = mds.fit_transform(dist_df.values)
    codes = list(dist_df.index)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, code in enumerate(codes):
        region = REGION_LABELS.get(code, "Other")
        color = REGION_COLORS.get(region, "gray")
        marker = "s" if region == "Eastern" else ("^" if region == "Other" else "o")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=120,
                   marker=marker, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(code, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, fontweight="bold", color=color)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=REGION_COLORS["Eastern"],
               label="Eastern Europe", markersize=10),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=REGION_COLORS["Western"],
               label="Western Europe", markersize=10),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=REGION_COLORS["Other"],
               label="Other (Israel)", markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_balance_comparison(props_df, path):
    """Bar chart comparing balance indices across countries by region."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("triangle_balance", "Triangle balance index (T)", axes[0, 0]),
        ("walk_balance", "Walk balance index (κ)", axes[0, 1]),
        ("signed_clustering", "Signed clustering coefficient (C)", axes[1, 0]),
        ("laplacian_balance", "Opposing Laplacian μ₁ (0 = balanced)", axes[1, 1]),
    ]

    df = props_df.sort_values("region")

    for col, title, ax in metrics:
        colors = [REGION_COLORS.get(r, "gray") for r in df["region"]]
        ax.bar(range(len(df)), df[col], color=colors, edgecolor="black",
               linewidth=0.3)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REGION_COLORS["Eastern"], label="Eastern"),
        Patch(facecolor=REGION_COLORS["Western"], label="Western"),
        Patch(facecolor=REGION_COLORS["Other"], label="Other"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Signed network balance properties by country",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_triad_profiles(props_df, path):
    """Stacked bar chart of signed triad census per country."""
    triad_cols = ["triad_+++", "triad_++-", "triad_+--", "triad_---"]
    triad_labels = ["+++", "++-", "+--", "---"]
    triad_colors = ["#2ca02c", "#98df8a", "#ff9896", "#d62728"]

    df = props_df.sort_values("region")

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(df))
    for col, label, color in zip(triad_cols, triad_labels, triad_colors):
        ax.bar(range(len(df)), df[col], bottom=bottom, label=label,
               color=color, edgecolor="black", linewidth=0.3)
        bottom += df[col].values

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of closed triads")
    ax.set_title("Signed triad census by country", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(title="Triad type", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_community_partitions(networks, beliefs, path):
    """Show the spectral bipartition of beliefs for each country."""
    codes = sorted(networks.keys())
    n_countries = len(codes)
    ncols = 6
    nrows = (n_countries + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for idx, code in enumerate(codes):
        ax = axes[idx]
        partition = spectral_partition(networks[code])
        colors = ["#1f77b4" if p == 0 else "#d62728" for p in partition]

        short_labels = [b.replace("anti_", "a_").replace("_", "\n")[:15]
                        for b in beliefs]

        y_pos = np.arange(len(beliefs))
        ax.barh(y_pos, partition * 2 - 1, color=colors, edgecolor="black",
                linewidth=0.3, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_labels, fontsize=5)
        region = REGION_LABELS.get(code, "Other")
        title_color = REGION_COLORS.get(region, "gray")
        ax.set_title(f"{code} ({COUNTRY_NAMES.get(code, code)})",
                     fontsize=9, fontweight="bold", color=title_color)
        ax.set_xlim(-1.5, 1.5)
        ax.axvline(0, color="black", linewidth=0.5)

    for idx in range(len(codes), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Spectral bipartition of beliefs (opposing Laplacian)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading networks...")
    networks, beliefs = load_all_networks()
    print(f"  Loaded {len(networks)} countries, {len(beliefs)} beliefs each\n")

    # --- Layer 1: Pairwise distances ---
    print("Computing pairwise distances...")
    frob_dist = frobenius_distance_matrix(networks)
    cos_sim = cosine_similarity_matrix(networks)
    corr_sim = correlation_similarity_matrix(networks)

    frob_dist.to_csv(OUT_DIR / "frobenius_distance.csv")
    cos_sim.to_csv(OUT_DIR / "cosine_similarity.csv")
    corr_sim.to_csv(OUT_DIR / "spearman_correlation.csv")

    plot_distance_heatmap(frob_dist, "Frobenius distance between belief networks",
                          OUT_DIR / "frobenius_heatmap.png")
    plot_similarity_heatmap(cos_sim, "Cosine similarity between belief networks",
                            OUT_DIR / "cosine_heatmap.png")
    plot_similarity_heatmap(corr_sim, "Spearman correlation between belief networks",
                            OUT_DIR / "spearman_heatmap.png")

    Z, codes = plot_dendrogram(frob_dist, "Hierarchical clustering of belief networks (Frobenius distance)",
                               OUT_DIR / "dendrogram_frobenius.png")
    save_dendrogram_csv(Z, codes, OUT_DIR / "dendrogram_linkage.csv")

    plot_mds(frob_dist, "MDS embedding of belief networks (Frobenius distance)",
             OUT_DIR / "mds_frobenius.png")

    # --- Layer 2: Structural properties ---
    print("Computing structural properties...")
    props = compute_all_properties(networks)
    props.to_csv(OUT_DIR / "network_properties.csv")

    plot_balance_comparison(props, OUT_DIR / "balance_comparison.png")
    plot_triad_profiles(props, OUT_DIR / "triad_profiles.png")

    # --- Layer 3: Community structure ---
    print("Computing community structure...")
    nmi_sim = community_similarity_matrix(networks)
    nmi_sim.to_csv(OUT_DIR / "community_nmi.csv")
    plot_similarity_heatmap(nmi_sim, "Community partition similarity (NMI)",
                            OUT_DIR / "community_nmi_heatmap.png")
    plot_community_partitions(networks, beliefs, OUT_DIR / "community_partitions.png")

    # --- Statistical tests ---
    print("\nStatistical tests (East vs. West)...")
    ew_labels = {c: REGION_LABELS[c] for c in frob_dist.index
                 if REGION_LABELS.get(c) in ("Eastern", "Western")}

    # Filter to East/West only
    ew_codes = [c for c in frob_dist.index if c in ew_labels]
    frob_ew = frob_dist.loc[ew_codes, ew_codes]

    diff, pval = mantel_test(frob_ew, ew_labels, n_perm=10000)
    print(f"  Frobenius distance:")
    print(f"    Mean between-group - mean within-group = {diff:.4f}")
    print(f"    Permutation p-value = {pval:.4f}")

    eastern_codes = [c for c in ew_codes if ew_labels[c] == "Eastern"]
    western_codes = [c for c in ew_codes if ew_labels[c] == "Western"]
    within_east = [frob_ew.loc[i, j] for i, j in combinations(eastern_codes, 2)]
    within_west = [frob_ew.loc[i, j] for i, j in combinations(western_codes, 2)]
    between = [frob_ew.loc[i, j] for i in eastern_codes for j in western_codes]

    print(f"\n  Mean within-Eastern distance:  {np.mean(within_east):.4f}")
    print(f"  Mean within-Western distance:  {np.mean(within_west):.4f}")
    print(f"  Mean between-group distance:   {np.mean(between):.4f}")

    # NMI-based test
    nmi_ew = nmi_sim.loc[ew_codes, ew_codes]
    nmi_dist = 1 - nmi_ew
    diff_nmi, pval_nmi = mantel_test(nmi_dist, ew_labels, n_perm=10000)
    print(f"\n  Community NMI distance:")
    print(f"    Mean between-group - mean within-group = {diff_nmi:.4f}")
    print(f"    Permutation p-value = {pval_nmi:.4f}")

    # --- Summary report ---
    report = []
    report.append("=" * 70)
    report.append("BELIEF NETWORK COMPARISON — SUMMARY REPORT")
    report.append("=" * 70)
    report.append(f"\nCountries: {len(networks)}")
    report.append(f"Beliefs: {len(beliefs)}")
    report.append(f"Eastern: {', '.join(sorted(EASTERN))}")
    report.append(f"Western: {', '.join(sorted(WESTERN))}")
    report.append(f"\n--- Layer 1: Edge-weight distances ---")
    report.append(f"Frobenius distance range: [{frob_dist.values[np.triu_indices_from(frob_dist.values, k=1)].min():.4f}, "
                   f"{frob_dist.values[np.triu_indices_from(frob_dist.values, k=1)].max():.4f}]")
    report.append(f"Cosine similarity range:  [{cos_sim.values[np.triu_indices_from(cos_sim.values, k=1)].min():.4f}, "
                   f"{cos_sim.values[np.triu_indices_from(cos_sim.values, k=1)].max():.4f}]")
    report.append(f"\n--- Layer 2: Structural properties ---")
    for col in ["triangle_balance", "walk_balance", "signed_clustering", "laplacian_balance"]:
        report.append(f"{col:25s}: mean={props[col].mean():.4f}, std={props[col].std():.4f}")
    report.append(f"\n--- Layer 3: Community structure ---")
    report.append(f"Mean NMI (all pairs): {nmi_sim.values[np.triu_indices_from(nmi_sim.values, k=1)].mean():.4f}")
    report.append(f"\n--- Statistical tests (East vs. West, excl. Israel) ---")
    report.append(f"Frobenius: diff={diff:.4f}, p={pval:.4f}")
    report.append(f"Community NMI: diff={diff_nmi:.4f}, p={pval_nmi:.4f}")
    report.append(f"\nMean within-Eastern:  {np.mean(within_east):.4f}")
    report.append(f"Mean within-Western:  {np.mean(within_west):.4f}")
    report.append(f"Mean between-group:   {np.mean(between):.4f}")
    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    with open(OUT_DIR / "summary_report.txt", "w") as f:
        f.write(report_text)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
