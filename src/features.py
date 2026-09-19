"""Feature construction for the hit-prediction model.

The governing rule: **every feature must be knowable before the game ships.**
A publisher deciding whether to greenlight a title has the platform, the
genre, the studio's track record, the ESRB rating and (close to launch) the
early review scores. It does not have the sales figures, the user review
count, or anything else the audience produces after release.

Two kinds of leakage are guarded against here:

*Target leakage* -- the regional sales columns sum to ``Global_Sales``, so any
of them turns the task into arithmetic. They never enter the feature matrix.

*Temporal leakage* -- track-record features like "how many hits has this
publisher had" are computed from strictly earlier release years, expanding
forward. Computing them over the whole dataset would let a 2004 row know
about 2012, and would quietly inflate every score in the evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Known before the game ships.
CATEGORICAL = ["Platform", "Genre", "Rating"]

NUMERIC = [
    "Year_of_Release",
    "platform_age",
    "is_sequel",
    "Critic_Score",
    "Critic_Count",
    "has_critic_score",
    "n_platforms_at_launch",
    "publisher_prior_titles",
    "publisher_prior_hits",
    "publisher_prior_hit_rate",
    "franchise_prior_titles",
    "franchise_prior_hits",
]

FEATURES = CATEGORICAL + NUMERIC

# Deliberately excluded, and used only in the leakage demonstration. Metacritic
# user review counts accumulate *after* release -- a game with 3,000 user
# reviews is a game lots of people already bought.
POST_RELEASE = ["User_Score", "User_Count", "has_user_score"]


def _expanding_track_record(
    df: pd.DataFrame, key: str, prefix: str
) -> pd.DataFrame:
    """Prior-year hit counts for each value of ``key``.

    For every row, returns how many titles that publisher (or franchise) had
    released, and how many of those were hits, in all years *strictly before*
    this row's release year. Rows from the first year a key appears get zeros,
    which is the honest answer: an unknown studio has no track record.
    """
    yearly = (
        df.groupby([key, "Year_of_Release"])
        .agg(titles=("is_hit", "size"), hits=("is_hit", "sum"))
        .reset_index()
        .sort_values([key, "Year_of_Release"])
    )

    # Cumulative totals then shifted by one year, so the current year's own
    # results are never part of its own track record.
    grouped = yearly.groupby(key, sort=False)
    yearly[f"{prefix}_prior_titles"] = grouped["titles"].cumsum() - yearly["titles"]
    yearly[f"{prefix}_prior_hits"] = grouped["hits"].cumsum() - yearly["hits"]

    return yearly[[key, "Year_of_Release",
                   f"{prefix}_prior_titles", f"{prefix}_prior_hits"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every engineered feature to a copy of the cleaned table."""
    d = df.copy()

    # Simultaneous multi-platform releases signal budget and expected reach,
    # and the SKU list is public before launch.
    d["n_platforms_at_launch"] = (
        d.groupby(["Name", "Year_of_Release"])["Platform"].transform("size")
    )

    for key, prefix in (("Publisher", "publisher"), ("franchise", "franchise")):
        track = _expanding_track_record(d, key, prefix)
        d = d.merge(track, on=[key, "Year_of_Release"], how="left")

    # A rate needs a denominator floor or a publisher with one prior release
    # and one prior hit reads as a guaranteed 100% success rate. Five is a
    # smoothing prior: it pulls thin track records toward the base rate.
    smoothing = 5.0
    base_rate = d["is_hit"].mean()
    d["publisher_prior_hit_rate"] = (
        (d["publisher_prior_hits"] + smoothing * base_rate)
        / (d["publisher_prior_titles"] + smoothing)
    )

    for col in ("publisher_prior_titles", "publisher_prior_hits",
                "franchise_prior_titles", "franchise_prior_hits"):
        d[col] = d[col].fillna(0)

    return d


def split_temporal(
    d: pd.DataFrame, train_end: int, test_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by release year: the past trains, the future tests.

    A random split would scatter a 2015 title's cross-platform siblings across
    both sides and let the model learn from the very slate it is being scored
    on. Time only runs one way in the real decision, so the validation should
    respect that.
    """
    train = d[d["Year_of_Release"] <= train_end]
    test = d[(d["Year_of_Release"] > train_end) & (d["Year_of_Release"] <= test_end)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def matrix(d: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Return just the model's input columns, in a fixed order."""
    cols = FEATURES + (extra or [])
    return d[cols].copy()
