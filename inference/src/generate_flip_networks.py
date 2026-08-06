"""Infer Peixoto belief networks for every alternate-coding dataset.

generate_alternate_coding.py flips 5 randomly chosen belief dimensions and
writes 100 samples per country (GB, RU) to:
    ../output/networks/random_flip/<CC>_5_dims/flipped_data_<i>.csv

This script fits a BeliefNetwork (PseudoNormalBlockState, Peixoto's MDL method)
on each of those datasets and saves the result under:
    ../output/networks/random_flip/networks/<CC>/<CC>_<i>.graphml
    ../output/networks/random_flip/networks/<CC>/<CC>_<i>.pkl

Parallelism note: graph-tool's OpenMP runtime deadlocks under Python
multiprocessing on macOS, so this script does NOT use multiprocessing. Instead
it processes a *shard* of the datasets serially in a single, single-threaded
process, and you launch several shards as independent OS processes (see the
launcher in run_flip_networks.sh). It is resumable: samples whose .graphml+.pkl
already exist are skipped.

Usage (from inference/src/):
    # one process, everything (slow, serial):
    python generate_flip_networks.py

    # shard k of n (launch n of these as separate processes):
    python generate_flip_networks.py --num-shards 12 --shard 0

    # build per-country summary.csv from already-saved networks:
    python generate_flip_networks.py --summarize
"""

import argparse
import glob
import os
import pickle
import re
import time

import pandas as pd

# Keep every process single-threaded; we parallelise across processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from belief_network import BeliefNetwork  # noqa: E402  (after env setup)

FLIP_DIR = "../output/networks/random_flip"
OUT_ROOT = "../output/networks/random_flip/networks"

BELIEF_DIMS = [
    "left_right_identification", "gender_inequality", "anti_lgbt",
    "euroscepticism", "anti_immigration", "anti_egalitarianism",
    "benefits_harm_economy", "benefits_harm_society", "welfare_chauvinism",
    "anti_economic_interventionism", "anti_social_benefits_low_income",
    "anti_social_benefits_parents", "educational_spending",
    "anti_basic_income", "anti_climate_change_taxes",
    "anti_climate_change_renewables", "anti_climate_ban_appliances",
    "climate_skepticism", "authoritarianism", "anti_libertarianism",
]


def discover_tasks(countries):
    """Find every (country, sample_index, csv_path), sorted deterministically."""
    tasks = []
    for cntry in countries:
        in_dir = os.path.join(FLIP_DIR, f"{cntry}_5_dims")
        if not os.path.isdir(in_dir):
            print(f"  (no directory for {cntry}: {in_dir}) — skipping")
            continue
        for csv_path in glob.glob(os.path.join(in_dir, "flipped_data_*.csv")):
            m = re.search(r"flipped_data_(\d+)\.csv$", os.path.basename(csv_path))
            if m:
                tasks.append((cntry, int(m.group(1)), csv_path))
    return sorted(tasks)


def fit_one(cntry, idx, csv_path):
    out_dir = os.path.join(OUT_ROOT, cntry)
    os.makedirs(out_dir, exist_ok=True)
    graphml_path = os.path.join(out_dir, f"{cntry}_{idx}.graphml")
    pkl_path = os.path.join(out_dir, f"{cntry}_{idx}.pkl")

    if os.path.exists(pkl_path) and os.path.exists(graphml_path):
        print(f"  {cntry} sample {idx:3d}  |  skipped (exists)", flush=True)
        return

    t0 = time.time()
    df = pd.read_csv(csv_path)
    bn = BeliefNetwork(BELIEF_DIMS)
    bn.fit(df, verbose=False)
    bn.save(graphml_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(bn, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  {cntry} sample {idx:3d}  |  n={len(df):,}  |  "
          f"{bn.graph.num_edges()} edges  |  DL={bn.state.entropy():.1f}  |  "
          f"{time.time() - t0:.1f}s", flush=True)


def summarize(countries):
    """Rebuild per-country summary.csv from saved .pkl networks."""
    for cntry in countries:
        out_dir = os.path.join(OUT_ROOT, cntry)
        rows = []
        for pkl_path in sorted(glob.glob(os.path.join(out_dir, f"{cntry}_*.pkl"))):
            m = re.search(rf"{cntry}_(\d+)\.pkl$", os.path.basename(pkl_path))
            if not m:
                continue
            with open(pkl_path, "rb") as f:
                bn = pickle.load(f)
            rows.append({
                "country": cntry,
                "sample": int(m.group(1)),
                "n_edges": bn.graph.num_edges(),
                "description_length": bn.state.entropy(),
            })
        if rows:
            rows.sort(key=lambda r: r["sample"])
            path = os.path.join(out_dir, "summary.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{cntry}: wrote {path} ({len(rows)} networks)")
        else:
            print(f"{cntry}: no networks found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=["GB", "RU"])
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--shard", type=int, default=0,
                        help="Which shard this process handles (0-indexed)")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip fitting; rebuild summary.csv from saved .pkl")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.countries)
        return

    tasks = discover_tasks(args.countries)
    my_tasks = tasks[args.shard::args.num_shards]  # deterministic modulo split
    print(f"[shard {args.shard}/{args.num_shards}] "
          f"{len(my_tasks)} of {len(tasks)} datasets "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})", flush=True)

    t0 = time.time()
    for cntry, idx, csv_path in my_tasks:
        fit_one(cntry, idx, csv_path)
    print(f"[shard {args.shard}/{args.num_shards}] done in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
