"""Round-specific configuration for ESS Round 8 data curation.

This module contains constants only. The principal output retains all
respondents aged 18 or older (and respondents with missing age), regardless
of belief missingness. The historical CCA threshold is retained only as an
eligibility flag and for published-reference validation. Shared cleaning,
rescaling, sample selection, weight handling, and validation functions belong
in ``ess_curation_common.py``.

The ordering of dictionaries and tuples is intentional because it determines
the final column order of the curated ESS8 datasets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Round identity and file names
# ---------------------------------------------------------------------------

ROUND_NUMBER = 8
ROUND_LABEL = "ESS Round 8"
RAW_DATA_FOLDER = "ESS8e02_3"
RAW_DATA_FILENAME = "ESS8e02_3.csv"
OUTPUT_PREFIX = "ess8"

OUTPUT_FILENAMES = {
    "without_weights": "ess8_beliefs_all_adults_without_weights.csv",
    "with_weights": "ess8_beliefs_all_adults_with_weights.csv",
    "validation": "ess8_cca_subset_validation_against_paper.csv",
}

# ---------------------------------------------------------------------------
# Expected dataset dimensions and sample definitions
# ---------------------------------------------------------------------------

EXPECTED_RAW_N = 44_387
EXPECTED_RAW_COUNTRIES = 23
EXPECTED_ANALYSIS_N = 43_148
EXPECTED_ANALYSIS_COUNTRIES = 23
EXPECTED_CCA_N = 37_118
EXPECTED_CCA_COUNTRIES = 23
EXPECTED_BELIEF_COUNT = 20

MINIMUM_AGE = 18
CCA_MAXIMUM_MISSING_BELIEFS = 2

# ---------------------------------------------------------------------------
# Official ESS survey weights
# ---------------------------------------------------------------------------

WEIGHT_COLUMNS = (
    "dweight",
    "pspwght",
    "pweight",
    "anweight",
)

# In the with-weights output, the weights are inserted immediately after this
# final metadata column and before the belief variables.
WEIGHT_INSERT_AFTER = "vote_simple"

# ---------------------------------------------------------------------------
# Country labels
# ---------------------------------------------------------------------------

COUNTRY_LABELS = {
    "AT": "Austria",
    "BE": "Belgium",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "DE": "Germany",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "HU": "Hungary",
    "IE": "Ireland",
    "IL": "Israel",
    "IS": "Iceland",
    "IT": "Italy",
    "LT": "Lithuania",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RU": "Russian Federation",
    "SE": "Sweden",
    "SI": "Slovenia",
}

# ---------------------------------------------------------------------------
# Metadata columns and raw-variable specifications
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS = (
    "ess_row_id",
    "idno",
    "cntry",
    "country_name",
    "ess_unique_id",
)

DEMOGRAPHIC_COLUMNS = (
    "agea",
    "gndr",
    "eisced",
    "education_3cat",
    "hinctnta",
    "rlgblg",
    "urbanization",
    "blgetmg",
    "vote_simple",
)

METADATA_COLUMNS_WITHOUT_WEIGHTS = IDENTIFIER_COLUMNS + DEMOGRAPHIC_COLUMNS
METADATA_COLUMNS_WITH_WEIGHTS = (
    METADATA_COLUMNS_WITHOUT_WEIGHTS
    + WEIGHT_COLUMNS
)

# Raw columns needed to construct identifiers, demographics, and voting
# metadata. The derived output names ``country_name``, ``ess_unique_id``,
# ``education_3cat``, ``urbanization``, and ``vote_simple`` are not raw names.
RAW_METADATA_COLUMNS = (
    "idno",
    "cntry",
    "agea",
    "gndr",
    "eisced",
    "hinctnta",
    "rlgblg",
    "domicil",
    "blgetmg",
    "vote",
)

# Valid substantive ranges for raw metadata variables.
METADATA_VALID_RANGES = {
    "agea": (0, 120),
    "gndr": (1, 2),
    "eisced": (1, 7),
    "hinctnta": (1, 10),
    "rlgblg": (1, 2),
    "domicil": (1, 5),
    "blgetmg": (1, 2),
    "vote": (1, 3),
}

# Collapsed ESS8 education categories based on ``eisced``.
EDUCATION_3CAT_MAP = {
    1: (1, 2),
    2: (3, 4, 5),
    3: (6, 7),
}

# The raw ``domicil`` scale is reversed: 1 = big city and 5 =
# farm/countryside. The derived ``urbanization`` variable is 6 - domicil,
# so higher values indicate a more urban setting.
URBANIZATION_REVERSE_CONSTANT = 6

# ---------------------------------------------------------------------------
# ESS8 belief construction
# ---------------------------------------------------------------------------

# Each belief is constructed as the strict row-wise mean of its listed items.
# A single missing constituent item makes the constructed belief missing.
BELIEF_MAP = {
    "left_right_identification": ("lrscale",),
    "gender_inequality": ("mnrgtjb",),
    "anti_lgbt": ("freehms", "hmsfmlsh", "hmsacld"),
    "euroscepticism": ("euftf",),
    "anti_immigration": ("imsmetn", "imdfetn", "impcntr"),
    "anti_egalitarianism": ("gincdif", "dfincac", "smdfslv"),
    "benefits_harm_economy": ("sbstrec", "sbbsntx"),
    "benefits_harm_society": ("sbprvpv", "sbeqsoc"),
    "welfare_chauvinism": ("imsclbn",),
    "anti_economic_interventionism": (
        "gvslvol",
        "gvslvue",
        "gvcldcr",
    ),
    "anti_social_benefits_low_income": ("bnlwinc",),
    "anti_social_benefits_parents": ("wrkprbf",),
    "educational_spending": ("eduunmp",),
    "anti_basic_income": ("basinc",),
    "anti_climate_change_taxes": ("inctxff",),
    "anti_climate_change_renewables": ("sbsrnen",),
    "anti_climate_ban_appliances": ("banhhap",),
    "climate_skepticism": ("ccnthum",),
    "authoritarianism": (
        "impsafe",
        "ipfrule",
        "ipbhprp",
        "ipstrgv",
        "imptrad",
    ),
    "anti_libertarianism": (
        "impdiff",
        "ipadvnt",
        "ipcrtiv",
        "impfree",
        "ipudrst",
    ),
}

BELIEF_COLUMNS = tuple(BELIEF_MAP)
BELIEF_ITEM_COLUMNS = tuple(
    dict.fromkeys(
        item
        for items in BELIEF_MAP.values()
        for item in items
    )
)

# Item-level coding instructions. Each value is:
#     (direction, valid minimum, valid maximum)
# where direction is either ``keep`` or ``reverse``.
# All final items are scaled to 0-1 and oriented so that higher values indicate
# more right-wing, conservative, or anti-progressive responses.
ITEM_CODING = {
    "lrscale": ("keep", 0, 10),
    "mnrgtjb": ("reverse", 1, 5),
    "freehms": ("keep", 1, 5),
    "hmsfmlsh": ("reverse", 1, 5),
    "hmsacld": ("keep", 1, 5),
    "euftf": ("reverse", 0, 10),
    "imsmetn": ("keep", 1, 4),
    "imdfetn": ("keep", 1, 4),
    "impcntr": ("keep", 1, 4),
    "gincdif": ("keep", 1, 5),
    "dfincac": ("reverse", 1, 5),
    "smdfslv": ("keep", 1, 5),
    "sbstrec": ("reverse", 1, 5),
    "sbbsntx": ("reverse", 1, 5),
    "sbprvpv": ("keep", 1, 5),
    "sbeqsoc": ("keep", 1, 5),
    "imsclbn": ("keep", 1, 5),
    "gvslvol": ("reverse", 0, 10),
    "gvslvue": ("reverse", 0, 10),
    "gvcldcr": ("reverse", 0, 10),
    "bnlwinc": ("reverse", 1, 4),
    "wrkprbf": ("reverse", 1, 4),
    "eduunmp": ("keep", 1, 4),
    "basinc": ("reverse", 1, 4),
    "inctxff": ("keep", 1, 5),
    "sbsrnen": ("keep", 1, 5),
    "banhhap": ("keep", 1, 5),
    "ccnthum": ("reverse", 1, 5),
    "impsafe": ("reverse", 1, 6),
    "ipfrule": ("reverse", 1, 6),
    "ipbhprp": ("reverse", 1, 6),
    "ipstrgv": ("reverse", 1, 6),
    "imptrad": ("reverse", 1, 6),
    "impdiff": ("keep", 1, 6),
    "ipadvnt": ("keep", 1, 6),
    "ipcrtiv": ("keep", 1, 6),
    "impfree": ("keep", 1, 6),
    "ipudrst": ("keep", 1, 6),
}

# All raw columns required by the ESS8 curation pipeline.
REQUIRED_RAW_COLUMNS = tuple(
    dict.fromkeys(
        RAW_METADATA_COLUMNS
        + WEIGHT_COLUMNS
        + BELIEF_ITEM_COLUMNS
    )
)

# ---------------------------------------------------------------------------
# Published Supplementary Table A2 validation targets
# ---------------------------------------------------------------------------

# Each row is: (belief variable, published N, published mean, published SD).
PAPER_STATISTICS = (
    ("left_right_identification", 34_248, 0.51, 0.22),
    ("gender_inequality", 37_038, 0.23, 0.27),
    ("anti_lgbt", 36_098, 0.34, 0.27),
    ("euroscepticism", 35_848, 0.51, 0.27),
    ("anti_immigration", 36_459, 0.45, 0.27),
    ("anti_egalitarianism", 36_772, 0.38, 0.19),
    ("benefits_harm_economy", 35_956, 0.49, 0.23),
    ("benefits_harm_society", 36_801, 0.41, 0.22),
    ("welfare_chauvinism", 36_441, 0.54, 0.26),
    ("anti_economic_interventionism", 36_927, 0.24, 0.16),
    ("anti_social_benefits_low_income", 36_399, 0.55, 0.27),
    ("anti_social_benefits_parents", 36_026, 0.47, 0.24),
    ("educational_spending", 36_227, 0.57, 0.25),
    ("anti_basic_income", 35_785, 0.50, 0.27),
    ("anti_climate_change_taxes", 36_832, 0.55, 0.31),
    ("anti_climate_change_renewables", 37_028, 0.26, 0.26),
    ("anti_climate_ban_appliances", 36_962, 0.36, 0.29),
    ("climate_skepticism", 36_035, 0.39, 0.20),
    ("authoritarianism", 36_631, 0.66, 0.17),
    ("anti_libertarianism", 36_769, 0.35, 0.16),
)

PAPER_STATISTICS_COLUMNS = (
    "belief_variable",
    "N_paper",
    "mean_paper",
    "sd_paper",
)

# ---------------------------------------------------------------------------
# Expected principal-output schemas
# ---------------------------------------------------------------------------

MISSINGNESS_COLUMNS = (
    "n_belief_missing",
    "n_belief_available",
    "cca_missingness_eligible",
)

OUTPUT_COLUMNS_WITHOUT_WEIGHTS = (
    METADATA_COLUMNS_WITHOUT_WEIGHTS
    + BELIEF_COLUMNS
    + MISSINGNESS_COLUMNS
)

OUTPUT_COLUMNS_WITH_WEIGHTS = (
    METADATA_COLUMNS_WITH_WEIGHTS
    + BELIEF_COLUMNS
    + MISSINGNESS_COLUMNS
)

EXPECTED_WITHOUT_WEIGHTS_SHAPE = (EXPECTED_ANALYSIS_N, 37)
EXPECTED_WITH_WEIGHTS_SHAPE = (EXPECTED_ANALYSIS_N, 41)
