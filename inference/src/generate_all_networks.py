"""Generate belief networks for all countries in the ESS8 data.

Usage (from inference/src/):
    python generate_all_networks.py

Reads : ../input/data/ESS8e02_3_self_processed/ess8_cca_initial_beliefs.csv
Writes: ../output/networks/<cntry>.graphml  (one per country)
        ../output/networks/summary.csv       (edge count + DL per country)
"""

import os
import sys
import time
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    countries = sorted(df["cntry"].unique())
    print(f"Loaded {len(df):,} respondents across {len(countries)} countries\n")

    summary = []

    for i, cntry in enumerate(countries, 1):
        df_c = df[df["cntry"] == cntry]
        n = len(df_c)
        print(f"[{i:2d}/{len(countries)}] {cntry}  (n={n:,})")

        t0 = time.time()
        bn = BeliefNetwork(BELIEF_DIMS)
        bn.fit(df_c, verbose=False)
        elapsed = time.time() - t0

        out_path = os.path.join(OUT_DIR, f"{cntry}.graphml")
        bn.save(out_path)

        n_edges = bn.graph.num_edges()
        dl = bn.state.entropy()
        print(f"         {n_edges} edges | DL = {dl:.1f} | {elapsed:.1f}s")

        summary.append({
            "country": cntry,
            "n_respondents": n,
            "n_edges": n_edges,
            "description_length": dl,
            "time_s": round(elapsed, 1),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDone. Networks saved to {OUT_DIR}/")
    print(f"Summary saved to {summary_path}")
    print(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
