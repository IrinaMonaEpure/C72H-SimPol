"""Export all GraphML belief networks to a single JSON for the webapp.

Includes node-level measures (degree centrality, strength) and edge weights.

Usage:
    python export_networks_json.py

Reads : ../output/networks/*.graphml
Writes: ../../webapp/public/networks.json
"""

import glob
import json
import os

import networkx as nx
import numpy as np


NETWORK_DIR = "../output/networks"
OUT_PATH = "../../webapp/public/networks.json"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czechia",
    "DE": "Germany", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "Great Britain", "HU": "Hungary", "IE": "Ireland",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
}


def load_graphml(path):
    G = nx.read_graphml(path)
    G_undirected = G.to_undirected()

    beliefs = nx.get_node_attributes(G, "belief")
    weights = nx.get_edge_attributes(G, "weight")

    node_list = list(G.nodes())
    label_map = {n: beliefs.get(n, n) for n in node_list}

    degree_cent = nx.degree_centrality(G_undirected)

    strength = {}
    for n in node_list:
        strength[n] = sum(
            abs(w) for _, _, w in G_undirected.edges(n, data="weight", default=0)
        )

    pos = nx.spring_layout(
        G_undirected, seed=42, k=1.2, iterations=500, weight="weight"
    )

    nodes = []
    for n in node_list:
        xy = pos[n]
        nodes.append({
            "id": label_map[n],
            "degree_centrality": round(degree_cent[n], 4),
            "node_strength": round(float(strength[n]), 4),
            "x": round(float(xy[0]), 6),
            "y": round(float(xy[1]), 6),
        })

    edges = []
    seen = set()
    for (u, v), w in weights.items():
        key = tuple(sorted([label_map[u], label_map[v]]))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source": label_map[u],
            "target": label_map[v],
            "weight": round(float(w), 4),
        })

    return {"nodes": nodes, "edges": edges}


def main():
    paths = sorted(glob.glob(os.path.join(NETWORK_DIR, "*.graphml")))
    print(f"Found {len(paths)} network files")

    data = {}
    for path in paths:
        code = os.path.splitext(os.path.basename(path))[0]
        name = COUNTRY_NAMES.get(code, code)
        print(f"  {code} ({name})")
        network = load_graphml(path)
        data[code] = {
            "name": name,
            "nodes": network["nodes"],
            "edges": network["edges"],
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
