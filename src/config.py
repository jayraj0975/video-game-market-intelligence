"""Project-wide paths and constants.

Every other module imports from here so that file locations and the
definitions of key business terms (what counts as a "hit", where the
train/test boundary sits) live in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_DIR / "video_game_sales.csv"
CLEAN_CSV = DATA_DIR / "video_game_sales_clean.csv"

for _d in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # read-only deployment (e.g. serverless); the app only reads

# ------------------------------------------------------------ constants
# A "hit" is a million-seller. This is the industry's own shorthand for a
# commercially successful title, which makes the model's output directly
# interpretable to a non-technical reader.
HIT_THRESHOLD_MUNITS = 1.0

# Sales are only meaningfully complete for this window. Pre-1996 rows are
# sparse and skewed toward a handful of Nintendo titles (fewer than 300 titles
# a year, almost no critic coverage), and 2017+ contains four stray rows that
# are plainly data-entry errors rather than a real release slate.
MIN_YEAR = 1996
MAX_YEAR = 2016

# 2016 is a partial year: the snapshot was taken in December 2016, so titles
# released that year had not finished selling. Total tracked sales fall to
# 130M units against 268M in 2015 on a comparable number of releases, which
# would depress the measured hit rate for reasons that have nothing to do with
# the games. It stays in the descriptive analysis (flagged) and is excluded
# from model evaluation, where it would silently penalise the model.
MODEL_MAX_YEAR = 2015

# Games released up to and including this year train the model; everything
# after it is held out. A random split would let the model see the future,
# which is exactly the thing a publisher cannot do.
TRAIN_END_YEAR = 2013

RANDOM_STATE = 42

# Columns that encode the target in disguise. Regional sales sum to
# Global_Sales, so including any of them turns the task into arithmetic.
LEAKY_COLUMNS = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
]

REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
REGION_LABELS = {
    "NA_Sales": "North America",
    "EU_Sales": "Europe",
    "JP_Sales": "Japan",
    "Other_Sales": "Rest of world",
}
