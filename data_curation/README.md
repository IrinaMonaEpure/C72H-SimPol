# ESS Round 4 and Round 8 data curation

This repository contains a reproducible Python workflow for curating European Social Survey (ESS) Round 4 (2008) and Round 8 (2016) data for subsequent correlational class analysis (CCA) and belief-network analysis.

The workflow reproduces the deterministic data-cleaning stage used by Jochem van Noord and colleagues and creates separate respondent-level datasets with and without the official ESS survey-weight columns.

## Main outputs

The pipeline creates four principal datasets:

```text
data/processed/
├── ess4_cca_initial_beliefs_without_weights.csv
├── ess4_cca_initial_beliefs_with_weights.csv
├── ess8_cca_initial_beliefs_without_weights.csv
└── ess8_cca_initial_beliefs_with_weights.csv
```

It also creates three validation and descriptive files:

```text
data/processed/
├── ess4_belief_validation_against_reference.csv
├── ess8_belief_validation_against_paper.csv
└── ess4_ess8_shared_belief_descriptives.csv
```

## Reproduced samples

| Round | Raw respondents | Final respondents | Countries | Constructed beliefs |
|---|---:|---:|---:|---:|
| ESS Round 4 | 56,752 | 45,268 | 29 | 19 |
| ESS Round 8 | 44,387 | 37,118 | 23 | 20 |

Respondents are retained when:

1. age is at least 18 years, or age is missing; and
2. no more than two constructed belief variables are missing.

Missing belief values are retained as missing and are not imputed.

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
│       ├── ess4_cca_initial_beliefs_without_weights.csv
│       ├── ess4_cca_initial_beliefs_with_weights.csv
│       ├── ess8_cca_initial_beliefs_without_weights.csv
│       ├── ess8_cca_initial_beliefs_with_weights.csv
│       ├── ess4_belief_validation_against_reference.csv
│       ├── ess8_belief_validation_against_paper.csv
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
└── README.md
```

## Workflow

### Notebook 1: ESS Round 8 curation

```text
notebooks/01_ess8_data_curation.ipynb
```

This notebook:

- reads `data/ESS8e02_3/ESS8e02_3.csv`;
- constructs identifiers, demographics, weights, and 20 belief variables;
- applies the age and missing-belief selection rules;
- creates separate with-weights and without-weights datasets;
- validates the belief-variable descriptive statistics against Supplementary Table A2.

### Notebook 2: ESS Round 4 curation

```text
notebooks/02_ess4_data_curation.ipynb
```

This notebook:

- reads `data/ESS4e04_6/ESS4e04_6.csv`;
- follows the supplied `Data cleaning_ESS4.R` definitions;
- constructs identifiers, demographics, weights, and 19 belief variables;
- applies the same age and missing-belief selection rules;
- creates separate with-weights and without-weights datasets.

### Notebook 3: validation and cross-round checks

```text
notebooks/03_validation_and_cross_round_checks.ipynb
```

This notebook:

- checks all four output schemas;
- verifies that weighted and unweighted versions differ only by the four official ESS weight columns;
- independently reproduces the ESS8 validation against Supplementary Table A2;
- compares the ESS4 Python output with Van Noord's saved `df_ESS4.RData`;
- verifies all 19 ESS4 belief variables respondent by respondent;
- summarizes the 12 belief concepts shared by ESS4 and ESS8.

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

To rebuild the datasets without opening the ESS4 R reference file:

```bash
python3 scripts/build_all_datasets.py --skip-ess4-reference-validation
```

Skipping the reference check is not recommended for the final reproducibility run.

## Official ESS weights

Both raw ESS files already contain the following official variables:

| Column | Meaning |
|---|---|
| `dweight` | Design weight |
| `pspwght` | Post-stratification weight including the design weight |
| `pweight` | Population-size weight |
| `anweight` | Combined analysis weight supplied by ESS |

The pipeline preserves these variables exactly as provided by ESS. It does not recalculate them.

The files named `with_weights` contain the four weight columns. The files named `without_weights` omit them. The respondents and belief values are otherwise identical.

The weight variables are metadata and must not be used as belief-network nodes. The appropriate weight depends on the downstream statistical estimand and analysis design.

## Belief variables

All constructed belief variables are scaled to the interval from 0 to 1. Higher values indicate the more right-wing, conservative, restrictive, or anti-progressive response according to the coding used in the source analysis.

ESS4 and ESS8 do not contain the same belief modules:

- ESS4 contains 19 constructed beliefs;
- ESS8 contains 20 constructed beliefs;
- 12 curated belief concepts are shared by both rounds.

The cross-round descriptive file is therefore restricted to those 12 shared concepts. It should not be interpreted as a respondent-level panel comparison because the rounds contain different respondents and different country coverage.

## Validation results

### ESS Round 8

The preprocessing reproduces:

- 37,118 retained respondents;
- 23 countries;
- all 20 published non-missing counts;
- all published means and standard deviations when rounded to two decimal places.

### ESS Round 4

The preprocessing reproduces:

- 45,268 retained respondents;
- 29 countries;
- the same respondent set as `df_ESS4.RData`;
- the same missingness pattern for all 19 beliefs;
- all 19 belief values respondent by respondent within numerical tolerance.

## Scope and limitations

This repository reproduces the deterministic data-curation stage and creates CCA-ready respondent-level belief data.

It does not yet reproduce:

- the modified stochastic CCA algorithm;
- the saved `countries_cca.RData` objects exactly, because the original random seed was not saved;
- later network, community-detection, regression, factor-analysis, or SEM results;
- the final seven-category ESS8 voting-behaviour variable derived from external PopuList classifications;
- a Round 4 voting-behaviour analysis, which was not included in the supplied Round 4 replication workflow.

The files named `with_weights` contain weight variables, but the original supplied CCA scripts did not apply ESS survey weights. A later weighted analysis must specify explicitly how the chosen method incorporates survey weights.

## Source analysis

This curation follows the data definitions and replication materials associated with:

Jochem van Noord, Felicity M. Turner-Zwinkels, Rebekka Kesberg, Mark J. Brandt, Matthew J. Easterbrook, Toon Kuppens, and Bram Spruyt, “The nature and structure of European belief systems: exploring the varieties of belief systems across 23 European countries”, *European Sociological Review*, 41(1):143–161, 2025.

DOI: https://doi.org/10.1093/esr/jcae011

Replication materials:

https://osf.io/hc3yz/
