"""Generate belief networks for all countries in the ESS8 data.

Usage (from inference/src/):
    python generate_all_networks.py

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
Writes: ../output/networks/<cntry>.graphml  (one per country)
        ../output/networks/<cntry>.pkl      (pickled BeliefNetwork object)
        ../output/networks/summary.csv       (edge count + DL per country)
"""

import os
import pickle
import time
from multiprocessing import Pool, cpu_count
import argparse

import pandas as pd
from belief_network import BeliefNetwork

DATA_PATH = "../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv"
OUT_DIR = "../output/networks"

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


def fit_country(args):
    cntry, df_c, flip = args
    n = len(df_c)
    out_dir = OUT_DIR if not flip else os.path.join(OUT_DIR, "flipped")
    t0 = time.time()

    bn = BeliefNetwork(BELIEF_DIMS)
    bn.fit(df_c, verbose=False)
    elapsed = time.time() - t0

    bn.save(os.path.join(out_dir, f"{cntry}.graphml"))

    pkl_path = os.path.join(out_dir, f"{cntry}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bn, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_edges = bn.graph.num_edges()
    dl = bn.state.entropy()
    print(f"  {cntry}  n={n:,}  |  {n_edges} edges  |  DL={dl:.1f}  |  {elapsed:.1f}s")

    return {
        "country": cntry,
        "n_respondents": n,
        "n_edges": n_edges,
        "description_length": dl,
        "time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip", action="store_true", help="Flip the coding of the belief dimensions")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.flip:
        out_dir = os.path.join(OUT_DIR, "flipped")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = OUT_DIR

    df = pd.read_csv(DATA_PATH)
    countries = sorted(df["cntry"].unique())
    n_workers = min(len(countries), cpu_count())
    print(f"Loaded {len(df):,} respondents across {len(countries)} countries")
    print(f"Running with {n_workers} parallel workers\n")

    if args.flip:
        for dim in BELIEF_DIMS:
            df[dim] = 1 - df[dim]
            tasks = [(cntry + f"_{dim}_flipped", df[df["cntry"] == cntry], args.flip) for cntry in countries]

            t_total = time.time()
            with Pool(n_workers) as pool:
                results = pool.map(fit_country, tasks)
            t_total = time.time() - t_total
            df[dim] = 1 - df[dim]  # revert the flip for the next dimension

            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(out_dir, "summary.csv")
            summary_df.to_csv(summary_path, index=False)

            print(f"\nDone in {t_total:.0f}s. Networks saved to {out_dir}/")
            print(f"Summary saved to {summary_path}")
            print(f"\n{summary_df.to_string(index=False)}")
    else:
        tasks = [(cntry, df[df["cntry"] == cntry], args.flip) for cntry in countries]

        t_total = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(fit_country, tasks)
        t_total = time.time() - t_total

        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(out_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\nDone in {t_total:.0f}s. Networks saved to {out_dir}/")
        print(f"Summary saved to {summary_path}")
        print(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
