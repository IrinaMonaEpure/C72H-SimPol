# ESS Round 4 and Round 8 data curation

This repository contains a reproducible Python workflow for curating European Social Survey (ESS) Round 4 (2008) and Round 8 (2016) data for subsequent belief-network and cross-round analyses.

The workflow reproduces the deterministic item cleaning and belief construction used by Jochem van Noord and colleagues, while revising the treatment of respondent-level missingness for the present project. The principal datasets retain all respondents aged 18 or older, as well as respondents whose age is missing, without excluding respondents because of the number of missing constructed beliefs.

Separate respondent-level datasets are created with and without the official ESS survey-weight columns.

## Main outputs

The pipeline creates four principal datasets:

```text
data/processed/
├── ess4_beliefs_all_adults_without_weights.csv
├── ess4_beliefs_all_adults_with_weights.csv
├── ess8_beliefs_all_adults_without_weights.csv
└── ess8_beliefs_all_adults_with_weights.csv
```

It also creates three validation and descriptive files:

```text
data/processed/
├── ess4_cca_subset_validation_against_reference.csv
├── ess8_cca_subset_validation_against_paper.csv
└── ess4_ess8_shared_belief_descriptives.csv
```

## Principal and historical samples

| Round | Raw respondents | Principal adult sample | Historical CCA-compatible subset | Countries | Constructed beliefs |
|---|---:|---:|---:|---:|---:|
| ESS Round 4 | 56,752 | 55,044 | 45,268 | 29 | 19 |
| ESS Round 8 | 44,387 | 43,148 | 37,118 | 23 | 20 |

The principal datasets retain respondents when:

1. age is at least 18 years; or
2. age is missing.

Respondents are **not excluded** because of the number of missing constructed belief variables.

This revision retains an additional:

- 9,776 ESS4 respondents with more than two missing beliefs;
- 6,030 ESS8 respondents with more than two missing beliefs.

The historical CCA-compatible subsets are retained only for validation against the earlier reference analysis. They contain respondents with no more than two missing constructed beliefs.

## Missing-value treatment

Invalid and non-substantive ESS responses, including values outside the valid response range for a variable, are converted to missing values (`NaN`).

For multi-item belief variables, a strict row mean is used. All constituent items must be available for the constructed belief to be non-missing. If one constituent item is missing, the corresponding constructed belief remains missing.

Missing belief values are:

- not imputed;
- not replaced by a mean, median, or other estimated value;
- retained in the principal respondent-level datasets.

Three diagnostic columns are included in every principal dataset:

| Column | Meaning |
|---|---|
| `n_belief_missing` | Number of missing constructed belief variables for the respondent |
| `n_belief_available` | Number of available constructed belief variables for the respondent |
| `cca_missingness_eligible` | `True` when the respondent has no more than two missing beliefs |

The `cca_missingness_eligible` variable does not determine inclusion in the principal datasets. It is included only so that the historical CCA-compatible subset can be reconstructed exactly for validation or sensitivity analyses.

## Project structure

```text
data_curation/
│
├── data/
│   ├── ESS4e04_6/
│   │   ├── ESS4e04_6.csv
│   │   └── ESS4e04_6 codebook.html
│   │
│   ├── ESS8e02_3/
│   │   ├── ESS8e02_3.csv
│   │   └── ESS8e02_3 codebook.html
│   │
│   └── processed/
│       ├── ess4_beliefs_all_adults_without_weights.csv
│       ├── ess4_beliefs_all_adults_with_weights.csv
│       ├── ess8_beliefs_all_adults_without_weights.csv
│       ├── ess8_beliefs_all_adults_with_weights.csv
│       ├── ess4_cca_subset_validation_against_reference.csv
│       ├── ess8_cca_subset_validation_against_paper.csv
│       └── ess4_ess8_shared_belief_descriptives.csv
│
├── notebooks/
│   ├── 01_ess8_data_curation.ipynb
│   ├── 02_ess4_data_curation.ipynb
│   └── 03_validation_and_cross_round_checks.ipynb
│
├── src/
│   ├── __init__.py
│   ├── ess_curation_common.py
│   ├── ess4_config.py
│   └── ess8_config.py
│
├── scripts/
│   └── build_all_datasets.py
│
├── reference/
│   ├── jcae011.pdf
│   ├── jcae011_suppl_supplementary_appendix.docx
│   └── van_noord/
│       ├── ESS Round 4/
│       ├── ESS Round 8/
│       └── index.txt
│
├── requirements.txt
└── README.md
```

## Workflow

### Notebook 1: ESS Round 8 curation

```text
notebooks/01_ess8_data_curation.ipynb
```

This notebook:

- reads `data/ESS8e02_3/ESS8e02_3.csv`;
- constructs identifiers, demographics, official weights, and 20 belief variables;
- retains respondents aged 18 or older and respondents whose age is missing;
- retains respondents regardless of the number of missing constructed beliefs;
- records the respondent-level missingness diagnostics and historical CCA-eligibility flag;
- creates separate with-weights and without-weights principal datasets;
- validates the historical 37,118-person CCA-compatible subset against Supplementary Table A2.

### Notebook 2: ESS Round 4 curation

```text
notebooks/02_ess4_data_curation.ipynb
```

This notebook:

- reads `data/ESS4e04_6/ESS4e04_6.csv`;
- follows the supplied `Data cleaning_ESS4.R` definitions;
- constructs identifiers, demographics, official weights, and 19 belief variables;
- retains respondents aged 18 or older and respondents whose age is missing;
- retains respondents regardless of the number of missing constructed beliefs;
- records the respondent-level missingness diagnostics and historical CCA-eligibility flag;
- creates separate with-weights and without-weights principal datasets.

### Notebook 3: validation and cross-round checks

```text
notebooks/03_validation_and_cross_round_checks.ipynb
```

This notebook:

- checks all four principal output schemas and sample sizes;
- verifies the adult-or-missing-age selection rule;
- confirms that respondents with more than two missing beliefs remain in the principal datasets;
- verifies that `n_belief_missing`, `n_belief_available`, and `cca_missingness_eligible` are internally consistent;
- verifies that weighted and unweighted versions differ only by the four official ESS weight columns;
- reconstructs the historical CCA-compatible subsets;
- validates the ESS8 historical subset against Supplementary Table A2;
- compares the ESS4 historical subset with Van Noord's saved `df_ESS4.RData`;
- verifies all 19 ESS4 belief variables respondent by respondent;
- summarizes the 12 belief concepts shared by ESS4 and ESS8 using the full principal adult samples.

## One-command reproducible build

After the raw and reference files are in the expected folders, run from the project root:

```bash
python3 scripts/build_all_datasets.py
```

The script rebuilds and validates all four principal datasets and all validation outputs.

The ESS4 reference check requires `pyreadr`. Install it with:

```bash
python3 -m pip install pyreadr
```

To rebuild the principal datasets without opening the ESS4 R reference file:

```bash
python3 scripts/build_all_datasets.py --skip-ess4-reference-validation
```

Skipping the ESS4 reference check is not recommended for the final reproducibility run.

## Official ESS weights

Both raw ESS files contain the following official variables:

| Column | Meaning |
|---|---|
| `dweight` | Design weight |
| `pspwght` | Post-stratification weight including the design weight |
| `pweight` | Population-size weight |
| `anweight` | Combined analysis weight supplied by ESS |

The pipeline preserves these variables as provided by ESS and does not recalculate them.

The files named `with_weights` contain the four weight columns. The files named `without_weights` omit them. The respondents, identifiers, demographics, belief values, and missingness diagnostics are otherwise identical.

The weight variables are metadata and must not be used as belief-network nodes. The appropriate weight depends on the downstream statistical estimand and analysis design.

## Belief variables

All constructed belief variables are scaled to the interval from 0 to 1. Higher values indicate the more right-wing, conservative, restrictive, or anti-progressive response according to the coding used in the source analysis.

ESS4 and ESS8 do not contain identical belief modules:

- ESS4 contains 19 constructed beliefs;
- ESS8 contains 20 constructed beliefs;
- 12 curated belief concepts are shared by both rounds.

The cross-round descriptive file is restricted to these 12 shared concepts and is calculated using the full principal adult samples.

It should not be interpreted as a respondent-level panel comparison because the two rounds contain different respondents and different country coverage.

## Validation results

### Principal ESS Round 8 dataset

The revised preprocessing produces:

- 43,148 principal adult respondents;
- 23 countries;
- 20 constructed belief variables;
- complete and internally consistent missingness diagnostics;
- matching weighted and unweighted respondent-level data.

The historical 37,118-person CCA-compatible subset reproduces:

- all 20 published non-missing counts;
- all published means and standard deviations when rounded to two decimal places.

### Principal ESS Round 4 dataset

The revised preprocessing produces:

- 55,044 principal adult respondents;
- 29 countries;
- 19 constructed belief variables;
- complete and internally consistent missingness diagnostics;
- matching weighted and unweighted respondent-level data.

The historical 45,268-person CCA-compatible subset reproduces:

- the same respondent set as `df_ESS4.RData`;
- the same missingness pattern for all 19 beliefs;
- all 19 belief values respondent by respondent within numerical tolerance.

## Scope and limitations

This repository reproduces the deterministic data-cleaning and belief-construction stages and creates respondent-level belief datasets for the present analysis.

It does not yet reproduce:

- the modified stochastic CCA algorithm;
- the saved `countries_cca.RData` objects exactly, because the original random seed was not saved;
- later network, community-detection, regression, factor-analysis, or structural-equation-modelling results;
- the final seven-category ESS8 voting-behaviour variable derived from external PopuList classifications;
- a Round 4 voting-behaviour analysis, which was not included in the supplied Round 4 replication workflow.

The historical CCA-compatible subsets are preserved only as reproducibility and sensitivity-analysis subsets. They are not the principal datasets used by the revised workflow.

The files named `with_weights` contain the survey-weight variables, but the original supplied CCA scripts did not apply ESS survey weights. Any later weighted analysis must specify explicitly how the chosen method incorporates survey weights.

## Source analysis

This curation follows the item definitions and replication materials associated with:

Jochem van Noord, Felicity M. Turner-Zwinkels, Rebekka Kesberg, Mark J. Brandt, Matthew J. Easterbrook, Toon Kuppens, and Bram Spruyt, “The nature and structure of European belief systems: exploring the varieties of belief systems across 23 European countries”, *European Sociological Review*, 41(1):143–161, 2025.

DOI: https://doi.org/10.1093/esr/jcae011

Replication materials:

https://osf.io/hc3yz/
