"""
Reproduce the 20-belief-variable dataset from Broere et al. (2022) using ESS Round 8 data.

Paper: "Belief systems and voting behaviour" (JCAE 2022, doi: 10.1093/jcae/jcae011)
Data: ESS8e02_3 (European Social Survey Round 8, 2016)

Processing steps (per paper's Appendix 2):
1. Load ESS8 data, keep only relevant items
2. Recode ESS missing-value codes to NaN; clip to theoretical range
3. Recode items so higher = right-wing/conservative
4. Normalise each item to [0, 1] using its theoretical scale range
5. Average normalised items within each of the 20 belief dimensions
   (strict: dim is NaN if ANY constituent item is NaN)
6. Keep respondents with at most 2 missing belief dimensions
   (paper: N increases from ~25,762 to 37,118; we obtain 37,659 — 1.5% off,
    likely due to minor differences in missing-value identification)
7. Save output with respondent IDs, country, weights, and 20 belief scores
"""

import pandas as pd
import numpy as np

ESS_PATH = "../input/ESS8e02_3/ESS8e02_3.csv"
OUT_PATH = "../output/ESS8_20beliefs_dummy.csv"

# ---------------------------------------------------------------------------
# Variable definitions
#   (dim_name, [(ess_item, reverse, scale_min, scale_max), ...])
#
#   reverse=True  → recode as (scale_max + scale_min - x) before normalising
#                   so that higher score = more right-wing for ALL items
#   Normalisation: (x_recoded - scale_min) / (scale_max - scale_min)
#
# Scale conventions (ESS):
#   lrscale  : 0–10 (0=left, 10=right) → keep
#   euftf    : 0–10 (0=unify more, 10=gone too far) → keep
#   imsmetn/imdfetn/impcntr : 1–4 (1=allow many, 4=allow none) → keep
#   imsclbn  : 1–4 (1=not entitled, 4=same as citizens) → reverse
#   gvslv*/gvcldcr : 0–10 (0=not govt resp, 10=govt resp) → reverse
#   bnlwinc/wrkprbf/eduunmp : 1–5 (1=much less, 5=much more) → reverse
#   basinc/inctxff/sbsrnen/banhhap : 1–5 (1=strongly favour, 5=strongly against) → keep
#   ccnthum  : 1–5 (1=not human, 5=definitely human) → reverse
#   values battery (impsafe,ipfrule,ipbhprp,ipstrgv,imptrad) : 1–6 (1=very much like me) → reverse
#   values battery (impdiff,ipadvnt,ipcrtiv,impfree,ipudrst) : 1–6 → keep (high=not like me=anti-libertarian)
#   mnrgtjb  : 1–5 (1=men more right, 5=women equal) → reverse
#   freehms/hmsfmlsh/hmsacld : 1–5 (1=agree=pro-LGBT, 5=disagree=anti-LGBT) → keep
#   gincdif  : 1–5 (1=agree govt should reduce diffs=egalitarian) → keep
#   dfincac  : 1–5 (1=not acceptable) — reverse so high=acceptable=right-wing
#   smdfslv  : 1–5 (1=disagree=not necessary) — reverse so high=agree=right-wing
#   sbprvpv  : 1–5 (1=disagree, 5=agree that benefits prevent work) → keep
#   sbeqsoc  : 1–5 (1=agree=benefits make more equal) → reverse
#   sbstrec  : 1–5 (1=disagree, 5=agree too great strain) → keep
#   sbbsntx  : 1–5 (1=disagree, 5=agree costs business) → keep
# ---------------------------------------------------------------------------

DIMS = [
    ("left_right",           [("lrscale",  False, 0, 10)]),
    ("gender_ineq",          [("mnrgtjb",  True,  1,  5)]),
    ("anti_lgbt",            [("freehms",  False, 1,  5),
                               ("hmsfmlsh", False, 1,  5),
                               ("hmsacld",  False, 1,  5)]),
    ("eurosceptic",          [("euftf",    False, 0, 10)]),
    ("anti_immigration",     [("imsmetn",  False, 1,  4),
                               ("imdfetn",  False, 1,  4),
                               ("impcntr",  False, 1,  4)]),
    ("anti_egalitarian",     [("gincdif",  False, 1,  5),
                               ("dfincac",  True,  1,  5),
                               ("smdfslv",  True,  1,  5)]),
    ("benefits_harm_economy",[("sbprvpv",  False, 1,  5),
                               ("sbeqsoc",  True,  1,  5)]),
    ("benefits_harm_society",[("sbstrec",  False, 1,  5),
                               ("sbbsntx",  False, 1,  5)]),
    ("welfare_chauvinism",   [("imsclbn",  True,  1,  4)]),
    ("anti_interventionism", [("gvslvol",  True,  0, 10),
                               ("gvslvue",  True,  0, 10),
                               ("gvcldcr",  True,  0, 10)]),
    ("anti_benefits_lowinc", [("bnlwinc",  True,  1,  5)]),
    ("anti_benefits_parents",[("wrkprbf",  True,  1,  5)]),
    ("anti_edu_spending",    [("eduunmp",  True,  1,  5)]),
    ("anti_basic_income",    [("basinc",   False, 1,  5)]),
    ("anti_climate_tax",     [("inctxff",  False, 1,  5)]),
    ("anti_climate_renew",   [("sbsrnen",  False, 1,  5)]),
    ("anti_climate_ban",     [("banhhap",  False, 1,  5)]),
    ("climate_skeptic",      [("ccnthum",  True,  1,  5)]),
    ("authoritarian",        [("impsafe",  True,  1,  6),
                               ("ipfrule",  True,  1,  6),
                               ("ipbhprp",  True,  1,  6),
                               ("ipstrgv",  True,  1,  6),
                               ("imptrad",  True,  1,  6)]),
    ("anti_libertarian",     [("impdiff",  False, 1,  6),
                               ("ipadvnt",  False, 1,  6),
                               ("ipcrtiv",  False, 1,  6),
                               ("impfree",  False, 1,  6),
                               ("ipudrst",  False, 1,  6)]),
]

# Missing-value codes by scale type (ESS convention)
MISS = {
    (0, 10): {77, 88, 99},
    (1,  6): {7, 8, 9},
    (1,  5): {6, 7, 8, 9},
    (1,  4): {5, 7, 8, 9},
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading ESS8 data …")
all_items = list(dict.fromkeys(item for _, items in DIMS for item, *_ in items))
weight_cols = ["dweight", "pspwght", "pweight", "anweight"]
df = pd.read_csv(ESS_PATH,
                 usecols=["idno", "cntry"] + weight_cols + all_items,
                 low_memory=False)
print(f"  {len(df):,} respondents, {sorted(df['cntry'].unique())}")

# ---------------------------------------------------------------------------
# Clean and normalise each item
# ---------------------------------------------------------------------------

print("Cleaning and normalising items …")
item_norm = {}  # item → normalised Series

for _, items in DIMS:
    for item, reverse, smin, smax in items:
        x = pd.to_numeric(df[item], errors="coerce")
        miss_set = MISS.get((smin, smax), set())
        if miss_set:
            x = x.where(~x.isin(miss_set), np.nan)
        x = x.where((x >= smin) & (x <= smax), np.nan)
        if reverse:
            x = (smax + smin) - x
        item_norm[item] = (x - smin) / (smax - smin)

# ---------------------------------------------------------------------------
# Compute 20 belief dimensions (strict: NaN if any item missing)
# ---------------------------------------------------------------------------

print("Computing belief dimensions …")
dim_cols = []
norm_df = pd.DataFrame(item_norm)

for dim, items in DIMS:
    icols = [item for item, *_ in items]
    # strict: skipna=False → NaN if any item is NaN
    df[dim] = norm_df[icols].mean(axis=1, skipna=False)
    dim_cols.append(dim)

# ---------------------------------------------------------------------------
# Filter: max 2 missing belief dimensions per respondent
# ---------------------------------------------------------------------------

n_miss = df[dim_cols].isnull().sum(axis=1)
keep = n_miss <= 2
df_out = df[keep].copy()
print(f"  Kept {keep.sum():,} / {len(df):,} respondents (≤2 missing dims)")
print(f"  (Paper reports 37,118; discrepancy ~1.5% likely from minor differences")
print(f"   in missing-value identification)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_cols = ["idno", "cntry"] + weight_cols + dim_cols
df_out[out_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"Shape: {df_out[out_cols].shape}")

print("\n--- Dimension means (higher = more right-wing) ---")
print(df_out[dim_cols].mean().round(3).to_string())

print("\n--- Missing counts per dimension ---")
print(df_out[dim_cols].isnull().sum().to_string())

print("\n--- Respondents per country ---")
print(df_out["cntry"].value_counts().sort_index().to_string())
