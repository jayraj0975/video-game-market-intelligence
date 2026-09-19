"""Load and clean the raw sales table.

The raw file is a VGChartz scrape joined to Metacritic. It has three problems
worth naming, all handled here:

1. ``User_Score`` is stored as text because Metacritic writes "tbd" for titles
   with too few user reviews to publish a score.
2. Sales are per platform release, not per game. "FIFA 15" appears five times.
   Some questions want the platform release (the unit a publisher actually
   ships); others want the title. Both views are produced.
3. Roughly half the rows have no critic score at all. That is not random --
   obscure titles go unreviewed -- so the missingness is kept as an explicit
   signal rather than imputed away.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (
    CLEAN_CSV,
    HIT_THRESHOLD_MUNITS,
    MAX_YEAR,
    MIN_YEAR,
    RAW_CSV,
)

# Words that mark a title as part of an ongoing series even when it carries no
# number. Matched case-insensitively against the title.
_SEQUEL_WORDS = (
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii", "xiii",
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "returns", "revenge", "encore", "remastered", "remake", "redux",
)


def _to_numeric_user_score(s: pd.Series) -> pd.Series:
    """Metacritic's 0-10 user score as a float, with 'tbd' as missing.

    Rescaled to the critic score's 0-100 range so the two are directly
    comparable in plots and in any model that sees both.
    """
    return pd.to_numeric(s.replace("tbd", np.nan), errors="coerce") * 10


def _looks_like_sequel(name: str) -> bool:
    """True when the title suggests an entry in an existing series.

    Deliberately crude. It is a proxy for "this game arrives with an audience
    already attached", and it only needs to be right often enough to carry
    signal, not right every time.
    """
    if not isinstance(name, str):
        return False
    tokens = re.split(r"[\s:\-]+", name.lower())
    return any(t in _SEQUEL_WORDS for t in tokens)


def _franchise_key(name: str) -> str:
    """A rough franchise identifier: the title up to its first colon.

    "Call of Duty: Black Ops" and "Call of Duty: Ghosts" collapse to
    "call of duty". Titles without a colon are their own franchise.
    """
    if not isinstance(name, str):
        return ""
    return re.split(r"[:\-]", name.lower(), maxsplit=1)[0].strip()


def _debut_years(df: pd.DataFrame, min_releases: int = 5) -> pd.Series:
    """First year each platform carried a real release slate.

    Robust to the handful of mislabelled release years in the raw scrape: a
    single stray row cannot pull a platform's debut back by two decades the
    way a plain ``min()`` would.
    """
    counts = df.groupby(["Platform", "Year_of_Release"]).size()
    real = counts[counts >= min_releases].reset_index()
    debut = real.groupby("Platform")["Year_of_Release"].min()
    # Platforms that never cleared the threshold in any single year fall back
    # to their earliest observed release.
    fallback = df.groupby("Platform")["Year_of_Release"].min()
    return debut.reindex(fallback.index).fillna(fallback).astype(int)


def _fully_observed(df: pd.DataFrame, decline_ratio: float = 0.5) -> pd.Series:
    """Platforms whose whole commercial life sits inside the analysis window.

    A platform qualifies when it launched after the window opened *and* its
    final observed year has fallen to less than ``decline_ratio`` of its peak
    -- that is, the decline has actually been observed. Without the second
    test the PS4 and Xbox One look like they peak in year two, when in fact
    the data simply stops in 2016 while they are still climbing.
    """
    yearly = df.groupby(["Platform", "Year_of_Release"])["Global_Sales"].sum()
    out = {}
    for platform, series in yearly.groupby(level=0):
        s = series.droplevel(0).sort_index()
        launched_in_window = df.loc[df["Platform"] == platform, "platform_debut"].iloc[0] >= MIN_YEAR
        declined = s.iloc[-1] <= decline_ratio * s.max()
        out[platform] = bool(launched_in_window and declined)
    return pd.Series(out)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as distributed, with no transformation."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python src/download_data.py` first."
        )
    return pd.read_csv(RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling-ready table.

    One row per platform release. Adds the target, the derived title features,
    and the explicit missingness flags.
    """
    d = df.copy()

    d = d.dropna(subset=["Name", "Genre"])
    d["Year_of_Release"] = pd.to_numeric(d["Year_of_Release"], errors="coerce")
    d = d.dropna(subset=["Year_of_Release"])

    # A platform's debut year has to come from the unfiltered table. Read it
    # after the 1996 cut and the PlayStation looks like it launched in 1996,
    # which would put every 1994-95 title at a negative platform age and make
    # the lifecycle curves quietly wrong.
    #
    # Taking the minimum year outright does not work either: the raw data has
    # stray mislabelled rows, one of which dates a Nintendo DS title to 1985
    # and would age every DS release by nineteen years. The debut is therefore
    # the first year the platform carried a real release slate.
    platform_debut = _debut_years(d, min_releases=5)

    d = d[d["Year_of_Release"].between(MIN_YEAR, MAX_YEAR)]
    d["Year_of_Release"] = d["Year_of_Release"].astype(int)

    d["User_Score"] = _to_numeric_user_score(d["User_Score"])
    d["Publisher"] = d["Publisher"].fillna("Unknown")
    d["Developer"] = d["Developer"].fillna("Unknown")
    d["Rating"] = d["Rating"].fillna("Unrated")

    # Missingness as signal, not noise: a title nobody reviewed is telling you
    # something about its commercial profile.
    d["has_critic_score"] = d["Critic_Score"].notna().astype(int)
    d["has_user_score"] = d["User_Score"].notna().astype(int)

    d["is_sequel"] = d["Name"].map(_looks_like_sequel).astype(int)
    d["franchise"] = d["Name"].map(_franchise_key)

    # How far into its life cycle the host platform was at release. Launch
    # titles and end-of-life titles sell very differently.
    d["platform_debut"] = d["Platform"].map(platform_debut).astype(int)
    d["platform_age"] = d["Year_of_Release"] - d["platform_debut"]

    # True when the platform's whole life is inside the analysis window. The
    # others are only partly observed, which makes their lifecycle curves
    # misleading even though their sales totals are fine.
    d["platform_fully_observed"] = d["Platform"].map(_fully_observed(d)).astype(int)

    d["is_hit"] = (d["Global_Sales"] >= HIT_THRESHOLD_MUNITS).astype(int)
    d["log_sales"] = np.log10(d["Global_Sales"])

    d = d.reset_index(drop=True)
    return d


def by_title(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse platform releases into one row per title.

    Sales are summed across platforms; scores are averaged; the release year
    is the earliest one. Use this when the question is about games rather than
    about SKUs -- "how many games sold a million" should not count a
    cross-platform title five times.
    """
    agg = {
        "Global_Sales": "sum",
        "NA_Sales": "sum",
        "EU_Sales": "sum",
        "JP_Sales": "sum",
        "Other_Sales": "sum",
        "Year_of_Release": "min",
        "Critic_Score": "mean",
        "User_Score": "mean",
        "Genre": "first",
        "Publisher": "first",
        "Platform": "nunique",
    }
    out = df.groupby("Name", as_index=False).agg(agg)
    out = out.rename(columns={"Platform": "n_platforms"})
    out["is_hit"] = (out["Global_Sales"] >= HIT_THRESHOLD_MUNITS).astype(int)
    return out


def build() -> pd.DataFrame:
    """Clean the raw file, cache the result, and return it."""
    d = clean(load_raw())
    d.to_csv(CLEAN_CSV, index=False)
    return d


def load_clean() -> pd.DataFrame:
    """Return the cleaned table, building it if the cache is absent."""
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    return build()


if __name__ == "__main__":
    d = build()
    print(f"cleaned rows      : {len(d):,}")
    print(f"unique titles     : {d['Name'].nunique():,}")
    print(f"years             : {d.Year_of_Release.min()}-{d.Year_of_Release.max()}")
    print(f"platforms         : {d.Platform.nunique()}")
    print(f"publishers        : {d.Publisher.nunique()}")
    print(f"hit rate          : {d.is_hit.mean():.1%}")
    print(f"critic score cover: {d.has_critic_score.mean():.1%}")
    print(f"saved             : {CLEAN_CSV}")
