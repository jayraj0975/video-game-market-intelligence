"""The guarantees the README makes about leakage, checked directly."""

import numpy as np
import pandas as pd

from features import build_features, split_temporal


def _slate() -> pd.DataFrame:
    rows = [
        # publisher, franchise, year, hit
        ("A", "f1", 2010, 1), ("A", "f1", 2010, 0),
        ("A", "f1", 2011, 1),
        ("A", "f2", 2012, 0),
        ("B", "f3", 2010, 0), ("B", "f3", 2012, 1),
        ("A", "f1", 2014, 1),
    ]
    return pd.DataFrame({
        "Name": [f"g{i}" for i in range(len(rows))],
        "Platform": "PS3",
        "Publisher": [r[0] for r in rows],
        "franchise": [r[1] for r in rows],
        "Year_of_Release": [r[2] for r in rows],
        "is_hit": [r[3] for r in rows],
    })


def test_track_record_uses_strictly_earlier_years():
    out = build_features(_slate())
    a2011 = out[(out.Publisher == "A") & (out.Year_of_Release == 2011)].iloc[0]
    # Publisher A released 2 titles (1 hit) in 2010 and nothing before that.
    assert a2011.publisher_prior_titles == 2
    assert a2011.publisher_prior_hits == 1
    # A row never sees its own year's results.
    a2010 = out[(out.Publisher == "A") & (out.Year_of_Release == 2010)]
    assert (a2010.publisher_prior_titles == 0).all()
    assert (a2010.publisher_prior_hits == 0).all()


def test_future_results_do_not_change_past_features():
    d = _slate()
    baseline = build_features(d, train_end=2012)
    flipped = d.copy()
    flipped.loc[flipped.Year_of_Release == 2014, "is_hit"] = 0
    changed = build_features(flipped, train_end=2012)
    cols = ["publisher_prior_titles", "publisher_prior_hits",
            "publisher_prior_hit_rate", "franchise_prior_hits"]
    early = baseline.Year_of_Release <= 2013
    pd.testing.assert_frame_equal(baseline.loc[early, cols], changed.loc[early, cols])


def test_smoothing_prior_ignores_held_out_years():
    d = _slate()
    flipped = d.copy()
    flipped.loc[flipped.Year_of_Release == 2014, "is_hit"] = 0
    a = build_features(d, train_end=2012)["publisher_prior_hit_rate"]
    b = build_features(flipped, train_end=2012)["publisher_prior_hit_rate"]
    # The 2014 row itself has the same history either way, and no row's rate
    # depends on the base rate measured over 2014.
    pd.testing.assert_series_equal(a, b)


def test_missing_publisher_gets_a_finite_rate():
    d = _slate()
    d.loc[0, "Publisher"] = np.nan
    out = build_features(d, train_end=2012)
    assert np.isfinite(out["publisher_prior_hit_rate"]).all()


def test_temporal_split_never_mixes_years():
    d = _slate()
    train, test = split_temporal(d, train_end=2012, test_end=2014)
    assert train.Year_of_Release.max() <= 2012 < test.Year_of_Release.min()
