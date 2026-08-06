# Script that generates alternate coding for a given number of dimensions chosen randomly. This is useful for testing the robustness of the network inference to coding choices.
import argparse
import os
import random
import pandas as pd

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

def flip_dimensions(df, dimensions):
    for dim in dimensions:
        if dim in df.columns:
            df[dim] = 1 - df[dim]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-dims", type=int, default=1, help="Number of dimensions to flip")
    parser.add_argument("--cntry", type=str, help="Country code to filter the data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of alternate coding samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating alternate coding for {args.num_dims} dimensions. Country filter: {args.cntry if args.cntry else 'None'}")

    # Load the original data
    df = pd.read_csv(DATA_PATH)

    # Check that the number of dimensions to flip is valid
    if args.num_dims < 1 or args.num_dims > len(BELIEF_DIMS):
        raise ValueError(f"num_dims must be between 1 and {len(BELIEF_DIMS)}")
    
    # Check that the specified country code is valid if provided
    if args.cntry is not None:
        if args.cntry not in df['cntry'].unique():
            raise ValueError(f"Country code {args.cntry} not found in the data")

    out_dir = os.path.join(OUT_DIR, "random_flip", f"{args.cntry}_{args.num_dims}_dims")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(out_dir, exist_ok=True)

    # Filter the data by country if specified
    if args.cntry:
        df = df[df['cntry'] == args.cntry]

    # Generate the specified number of samples of alternate coding for the specified number of dimensions
    for i in range(args.num_samples):
        # Randomly select dimensions to flip
        selected_dims = random.sample(BELIEF_DIMS, min(args.num_dims, len(BELIEF_DIMS)))

        # Save the selected dimensions in a csv file for reference
        dims_path = os.path.join(out_dir, f"flipped_dims_{i}.csv")
        pd.DataFrame(selected_dims, columns=["dimension"]).to_csv(dims_path, index=False)

        # Flip the selected dimensions
        flipped_df = flip_dimensions(df.copy(), selected_dims)

        # Save the flipped data
        output_path = os.path.join(out_dir, f"flipped_data_{i}.csv")
        flipped_df.to_csv(output_path, index=False)
        print(f"Flipped data saved to {output_path}. Flipped dimensions: {selected_dims}")

if __name__ == "__main__":
    main()