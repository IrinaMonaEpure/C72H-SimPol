"""Round-specific configuration for ESS Round 4 data curation.

This module translates the deterministic item construction choices in Jochem
van Noord's supplied ``Data cleaning_ESS4.R`` script into explicit Python
configuration. The principal output retains all respondents aged 18 or older
(and respondents with missing age), regardless of belief missingness. The
historical CCA threshold is retained only as an eligibility flag and for
reference validation. Shared cleaning, rescaling, sample selection, weight
handling, and validation functions belong in ``ess_curation_common.py``.

The ordering of dictionaries and tuples is intentional because it determines
the final column order of the curated ESS4 datasets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Round identity and file names
# ---------------------------------------------------------------------------

ROUND_NUMBER = 4
ROUND_LABEL = "ESS Round 4"
RAW_DATA_FOLDER = "ESS4e04_6"
RAW_DATA_FILENAME = "ESS4e04_6.csv"
OUTPUT_PREFIX = "ess4"

OUTPUT_FILENAMES = {
    "without_weights": "ess4_beliefs_all_adults_without_weights.csv",
    "with_weights": "ess4_beliefs_all_adults_with_weights.csv",
    "validation": "ess4_cca_subset_validation_against_reference.csv",
}

# Van Noord reference files retained under reference/van_noord/.
REFERENCE_CLEANING_SCRIPT_PARTS = (
    "reference",
    "van_noord",
    "ESS Round 4",
    "scripts",
    "Data cleaning_ESS4.R",
)
REFERENCE_RDATA_PARTS = (
    "reference",
    "van_noord",
    "ESS Round 4",
    "data",
    "df_ESS4.RData",
)

# ---------------------------------------------------------------------------
# Expected dataset dimensions and sample definitions
# ---------------------------------------------------------------------------

EXPECTED_RAW_N = 56_752
EXPECTED_RAW_COUNTRIES = 29
EXPECTED_ANALYSIS_N = 55_044
EXPECTED_ANALYSIS_COUNTRIES = 29
EXPECTED_CCA_N = 45_268
EXPECTED_CCA_COUNTRIES = 29
EXPECTED_BELIEF_COUNT = 19

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

# Round 4 has no vote-behaviour field in Van Noord's curation. Insert the
# weights after the final demographic column.
WEIGHT_INSERT_AFTER = "blgetmg"

# ---------------------------------------------------------------------------
# Country labels
# ---------------------------------------------------------------------------

COUNTRY_LABELS = {
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CH": "Switzerland",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IL": "Israel",
    "LV": "Latvia",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RU": "Russian Federation",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "TR": "Türkiye",
    "UA": "Ukraine",
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
    "edulvla",
    "education_3cat",
    "hinctnta",
    "rlgblg",
    "urbanization",
    "blgetmg",
)

METADATA_COLUMNS_WITHOUT_WEIGHTS = IDENTIFIER_COLUMNS + DEMOGRAPHIC_COLUMNS
METADATA_COLUMNS_WITH_WEIGHTS = METADATA_COLUMNS_WITHOUT_WEIGHTS + WEIGHT_COLUMNS

# Derived output names ``country_name``, ``ess_unique_id``,
# ``education_3cat``, and ``urbanization`` are not raw ESS columns.
RAW_METADATA_COLUMNS = (
    "idno",
    "cntry",
    "agea",
    "gndr",
    "edulvla",
    "hinctnta",
    "rlgblg",
    "domicil",
    "blgetmg",
)

# Valid substantive ranges. ESS non-response codes outside these intervals are
# converted to missing values before further processing.
METADATA_VALID_RANGES = {
    "agea": (15, 123),
    "gndr": (1, 2),
    "edulvla": (1, 5),
    "hinctnta": (1, 10),
    "rlgblg": (1, 2),
    "domicil": (1, 5),
    "blgetmg": (1, 2),
}

# Van Noord's Round 4 three-category education recoding based on ``edulvla``.
EDUCATION_3CAT_MAP = {
    1: (1, 2),
    2: (3, 4),
    3: (5,),
}

# Raw ``domicil`` is 1 = big city through 5 = farm/countryside. The derived
# ``urbanization`` variable is 6 - domicil, so higher values mean more urban.
URBANIZATION_REVERSE_CONSTANT = 6

# ---------------------------------------------------------------------------
# ESS4 belief construction
# ---------------------------------------------------------------------------

# The 19 final beliefs follow the supplied Round 4 cleaning script. Common
# concepts use the same descriptive names as the ESS8 output. Each multi-item
# belief uses a strict row mean: a missing constituent item makes the entire
# constructed belief missing.
BELIEF_MAP = {
    "left_right_identification": ("lrscale",),
    "gender_inequality": ("mnrgtjb", "wmcpwrk"),
    "anti_lgbt": ("freehms",),
    "euroscepticism": ("euftf",),
    "anti_immigration": ("imsmetn", "imdfetn", "impcntr"),
    "anti_egalitarianism": ("gincdif", "smdfslv"),
    "benefits_harm_economy": ("sbstrec", "sbbsntx"),
    "benefits_harm_society": ("sbprvpv", "sbeqsoc"),
    "welfare_chauvinism": ("imsclbn",),
    "anti_economic_interventionism": (
        "gvslvol",
        "gvslvue",
        "gvcldcr",
        "gvjbevn",
        "gvhlthc",
        "gvpdlwk",
    ),
    "harsh_sentences": ("hrshsnt",),
    "anti_militant_democracy": ("prtyban",),
    "no_science_environment_solution": ("scnsenv",),
    "anti_government_spending": ("ditxssp",),
    "regressive_taxes": ("txearn",),
    "regressive_benefits": ("earnpen", "earnueb"),
    "age_prejudice": ("buproag",),
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

# Item-level coding instructions applied after any special value recoding.
# Each value is: (direction, valid minimum, valid maximum), where direction is
# either ``keep`` or ``reverse``. Final items are placed on a 0-1 scale.
ITEM_CODING = {
    "lrscale": ("keep", 0, 10),
    "mnrgtjb": ("keep", 1, 5),
    "wmcpwrk": ("keep", 1, 5),
    "freehms": ("keep", 1, 5),
    "euftf": ("reverse", 0, 10),
    "imsmetn": ("keep", 1, 4),
    "imdfetn": ("keep", 1, 4),
    "impcntr": ("keep", 1, 4),
    "gincdif": ("keep", 1, 5),
    "smdfslv": ("keep", 1, 5),
    "sbstrec": ("reverse", 1, 5),
    "sbbsntx": ("reverse", 1, 5),
    "sbprvpv": ("keep", 1, 5),
    "sbeqsoc": ("keep", 1, 5),
    "imsclbn": ("keep", 1, 5),
    "gvslvol": ("reverse", 0, 10),
    "gvslvue": ("reverse", 0, 10),
    "gvcldcr": ("reverse", 0, 10),
    "gvjbevn": ("reverse", 0, 10),
    "gvhlthc": ("reverse", 0, 10),
    "gvpdlwk": ("reverse", 0, 10),
    "hrshsnt": ("reverse", 1, 5),
    "prtyban": ("keep", 1, 5),
    "scnsenv": ("keep", 1, 5),
    "ditxssp": ("reverse", 0, 10),
    "txearn": ("keep", 1, 3),
    "earnpen": ("keep", 1, 3),
    "earnueb": ("keep", 1, 3),
    "buproag": ("reverse", 0, 10),
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

# Special recoding performed in the supplied R script before 0-1 scaling.
# Values not listed in the mapping become missing.
ITEM_VALUE_RECODING = {
    "txearn": {
        1: 2,
        2: 1,
        3: 3,
    },
}

# The R script explicitly treats response 4 as missing for these two items.
# Their ITEM_CODING range (1-3) also enforces this rule in Python.
ITEM_VALUES_FORCED_MISSING = {
    "earnpen": (4,),
    "earnueb": (4,),
}

# ``dfincac`` was selected and examined in the original R script but was
# deliberately excluded from the final anti-egalitarianism scale. It is
# therefore not required by the Python curation pipeline.
EXAMINED_BUT_EXCLUDED_ITEMS = ("dfincac",)

REQUIRED_RAW_COLUMNS = tuple(
    dict.fromkeys(
        RAW_METADATA_COLUMNS
        + WEIGHT_COLUMNS
        + BELIEF_ITEM_COLUMNS
    )
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

EXPECTED_WITHOUT_WEIGHTS_SHAPE = (EXPECTED_ANALYSIS_N, 35)
EXPECTED_WITH_WEIGHTS_SHAPE = (EXPECTED_ANALYSIS_N, 39)
