"""The trained model and the lookups the API needs, built once and cached.

The served model is the same one the model report evaluates: logistic
regression trained on 1996-2013 releases with isotonic calibration, so the
probabilities it returns are honest frequencies rather than balanced-class
scores. Publisher and franchise track records come from every release the
dataset covers (through 2015), since a new title is judged on all history.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DATA_DIR, MODEL_MAX_YEAR, REPORTS_DIR, TRAIN_END_YEAR  # noqa: E402
from data_prep import _franchise_key, _looks_like_sequel, load_clean  # noqa: E402
from features import FEATURES, NUMERIC, build_features, split_temporal  # noqa: E402
from train import logistic_pipeline  # noqa: E402

MODEL_PATH = DATA_DIR / "serving_model.joblib"
SMOOTHING = 5.0  # must match features.build_features


@dataclass
class Artifacts:
    model: object
    base_rate: float
    platforms: list[str]
    genres: list[str]
    ratings: list[str]
    platform_debut: dict[str, int]
    publishers: dict[str, tuple[int, int]]   # name -> (titles, hits)
    franchises: dict[str, tuple[int, int]]   # key  -> (titles, hits)
    market: dict


def _history(d: pd.DataFrame, key: str) -> dict[str, tuple[int, int]]:
    g = d.groupby(key)["is_hit"].agg(["size", "sum"])
    return {k: (int(r["size"]), int(r["sum"])) for k, r in g.iterrows()}


def _market_summary(d: pd.DataFrame) -> dict:
    """Small, chart-ready aggregates for the dashboard."""
    regions = {"North America": "NA_Sales", "Europe": "EU_Sales", "Japan": "JP_Sales"}
    share = {}
    for label, col in regions.items():
        s = d.groupby("Genre")[col].sum()
        share[label] = {g: float(v) for g, v in (s / s.sum()).items()}
    scored = d.dropna(subset=["Critic_Score"]).copy()
    bins = [0, 50, 60, 70, 80, 90, 101]
    names = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]
    scored["band"] = pd.cut(scored["Critic_Score"], bins=bins, labels=names, right=False)
    hit_rate = scored.groupby("band", observed=True)["is_hit"].agg(["mean", "size"])
    return {
        "regional_genre_share": share,
        "hit_rate_by_score": [
            {"band": b, "hit_rate": float(hit_rate.loc[b, "mean"]),
             "titles": int(hit_rate.loc[b, "size"])}
            for b in names if b in hit_rate.index
        ],
    }


def build_artifacts() -> Artifacts:
    d = build_features(load_clean(), train_end=TRAIN_END_YEAR)
    train, _ = split_temporal(d, TRAIN_END_YEAR, MODEL_MAX_YEAR)

    model = CalibratedClassifierCV(
        clone(logistic_pipeline(NUMERIC)), method="isotonic", cv=5,
    ).fit(train[FEATURES], train["is_hit"].to_numpy())

    known = d[d["Year_of_Release"] <= MODEL_MAX_YEAR]
    return Artifacts(
        model=model,
        base_rate=float(train["is_hit"].mean()),
        platforms=sorted(d["Platform"].unique()),
        genres=sorted(d["Genre"].unique()),
        ratings=sorted(d["Rating"].unique()),
        platform_debut={k: int(v) for k, v in d.groupby("Platform")["platform_debut"].first().items()},
        publishers=_history(known, "Publisher"),
        franchises=_history(known, "franchise"),
        market=_market_summary(known),
    )


_cache: Artifacts | None = None


def get_artifacts() -> Artifacts:
    """Load from disk if present, otherwise train and cache."""
    global _cache
    if _cache is None:
        if MODEL_PATH.exists():
            try:
                _cache = joblib.load(MODEL_PATH)
            except Exception:
                _cache = None
        if _cache is None:
            _cache = build_artifacts()
            joblib.dump(_cache, MODEL_PATH)
    return _cache


def feature_frame(a: Artifacts, items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        p_titles, p_hits = a.publishers.get(it["publisher"], (0, 0))
        key = _franchise_key(it["title"]) if it.get("title") else ""
        f_titles, f_hits = a.franchises.get(key, (0, 0))
        score, count = it.get("critic_score"), it.get("critic_count")
        rows.append({
            "Platform": it["platform"],
            "Genre": it["genre"],
            "Rating": it["rating"],
            "Year_of_Release": it["year"],
            "platform_age": max(0, it["year"] - a.platform_debut[it["platform"]]),
            "is_sequel": int(_looks_like_sequel(it["title"])) if it.get("title") else 0,
            "Critic_Score": np.nan if score is None else score,
            "Critic_Count": np.nan if count is None else count,
            "has_critic_score": int(score is not None),
            "n_platforms_at_launch": it.get("n_platforms", 1),
            "publisher_prior_titles": p_titles,
            "publisher_prior_hits": p_hits,
            "publisher_prior_hit_rate": (p_hits + SMOOTHING * a.base_rate) / (p_titles + SMOOTHING),
            "franchise_prior_titles": f_titles,
            "franchise_prior_hits": f_hits,
        })
    return pd.DataFrame(rows)[FEATURES]


def predict(a: Artifacts, items: list[dict]) -> list[dict]:
    # Isotonic calibration saturates at exactly 0 and 1 on the extremes, which
    # is a step-function artefact rather than a certainty. Clip to a range the
    # 14k-row training set can actually support.
    probs = np.clip(a.model.predict_proba(feature_frame(a, items))[:, 1], 0.005, 0.98)
    out = []
    for it, p in zip(items, probs):
        p_titles, p_hits = a.publishers.get(it["publisher"], (0, 0))
        key = _franchise_key(it["title"]) if it.get("title") else ""
        f_titles, f_hits = a.franchises.get(key, (0, 0))
        out.append({
            "title": it.get("title") or "",
            "platform": it["platform"],
            "genre": it["genre"],
            "probability": float(p),
            "vs_base_rate": float(p / a.base_rate),
            "publisher_history": {"titles": p_titles, "hits": p_hits},
            "franchise_history": {"titles": f_titles, "hits": f_hits},
            "used_critic_data": it.get("critic_score") is not None,
        })
    return out


def load_metrics() -> dict:
    import json
    return json.loads((REPORTS_DIR / "metrics.json").read_text())
