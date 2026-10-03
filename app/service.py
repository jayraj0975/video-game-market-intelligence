"""The trained model and the lookups the API needs, built once and cached.

The served model is the same one the model report evaluates: logistic
regression trained on 1996-2013 releases with isotonic calibration, so the
probabilities it returns are honest frequencies rather than balanced-class
scores. Publisher and franchise track records come from every release the
dataset covers (through 2015), since a new title is judged on all history. That
history is only a valid input for releases AFTER 2015; scoring an older release
lets the model see results from its own future, and the API says so.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    CLEAN_CSV,
    DECISION_POINT_LABEL,
    DECISION_POINT_NOTE,
    MODEL_MAX_YEAR,
    MODEL_VERSION,
    RAW_CSV,
    REPORTS_DIR,
    TRAIN_END_YEAR,
)
from data_prep import _franchise_key, _looks_like_sequel  # noqa: E402
from features import FEATURES, NUMERIC  # noqa: E402

# Committed to the repository so a deployment needs neither the dataset nor a
# training step. Rebuild with ``python -m app.service`` after changing the model.
MODEL_PATH = Path(__file__).parent / "model" / "serving_model.joblib"
# Provenance for the artifact above: version, commit, data hashes, windows, artifact hash.
META_PATH = MODEL_PATH.with_name("serving_model.meta.json")
# Publisher and franchise track records are built from releases through this year.
HISTORY_THROUGH = MODEL_MAX_YEAR
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
    catalog: list[dict]   # past releases (through HISTORY_THROUGH) for comparables lookup


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
    # Imported here so serving never pulls in the plotting stack or needs data.
    from data_prep import load_clean
    from features import build_features, split_temporal
    from train import logistic_pipeline

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
        catalog=_catalog(known),
    )


def _catalog(known: pd.DataFrame) -> list[dict]:
    """A slim record per past release, for 'what happened to games like this'."""
    cat = known[["Name", "Platform", "Genre", "Rating", "Publisher",
                 "Year_of_Release", "Critic_Score", "Global_Sales", "is_hit"]].copy()
    cat["Critic_Score"] = cat["Critic_Score"].astype(object).where(cat["Critic_Score"].notna(), None)
    cat = cat.rename(columns={
        "Name": "title", "Platform": "platform", "Genre": "genre", "Rating": "rating",
        "Publisher": "publisher", "Year_of_Release": "year", "Critic_Score": "critic_score",
        "Global_Sales": "global_sales_munits",
    })
    return cat.to_dict("records")


_cache: Artifacts | None = None


def get_artifacts() -> Artifacts:
    """Load from disk if present, otherwise train and cache."""
    global _cache
    if _cache is None:
        if MODEL_PATH.exists():
            _cache = Artifacts(**joblib.load(MODEL_PATH))
        else:
            _cache = build_artifacts()
            save_artifacts(_cache)
    return _cache


def save_artifacts(a: Artifacts) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A plain dict, so loading does not depend on where this module is imported from.
    joblib.dump(asdict(a), MODEL_PATH, compress=3)


def write_meta(a: Artifacts) -> dict:
    """Describe the artifact just saved (hashes, versions, windows) in a JSON file beside it."""
    from provenance import describe, sha256_file

    meta = describe(FEATURES, data_files={"raw_csv": RAW_CSV, "clean_csv": CLEAN_CSV}, extra={
        "model_version": MODEL_VERSION,
        "decision_point": DECISION_POINT_LABEL,
        "decision_point_note": DECISION_POINT_NOTE,
        "algorithm": "Logistic regression (class-weighted), isotonic calibration (5-fold)",
        "training_window": f"1996-{TRAIN_END_YEAR}",
        "evaluation_window": f"{TRAIN_END_YEAR + 1}-{MODEL_MAX_YEAR} (reported in reports/metrics.json)",
        "history_through": HISTORY_THROUGH,
        "base_rate": a.base_rate,
        "artifact_file": MODEL_PATH.name,
        "artifact_sha256": sha256_file(MODEL_PATH),
        "artifact_bytes": MODEL_PATH.stat().st_size,
    })
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_meta() -> dict | None:
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else None


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
            "decision_point": DECISION_POINT_LABEL,
            "history_through": HISTORY_THROUGH,
            # Track records include releases up to HISTORY_THROUGH, so a release dated on or before
            # that year is being scored with knowledge of its own future.
            "history_note": (
                f"Publisher and franchise track records include releases through {HISTORY_THROUGH}, "
                f"later than {it['year']}, so this is not a valid historical backtest. Score new releases."
                if it["year"] <= HISTORY_THROUGH else None
            ),
        })
    return out


def _comparable_score_breakdown(row: dict, item: dict) -> dict:
    """Per-component contributions to the similarity score, so a caller can show
    *why* two releases matched instead of just the total (see ``explain_comparable``
    in ai.py). ``total`` is exactly what ``_comparable_score`` returns.
    """
    platform_match = 3.0 if row["platform"] == item["platform"] else 0.0
    genre_match = 3.0 if row["genre"] == item["genre"] else 0.0
    rating_match = 1.0 if row["rating"] == item["rating"] else 0.0
    publisher_match = 2.0 if item.get("publisher") and row["publisher"] == item["publisher"] else 0.0
    critic_closeness = (
        max(0.0, 2.0 - abs(row["critic_score"] - item["critic_score"]) / 10)
        if item.get("critic_score") is not None and row["critic_score"] is not None else 0.0
    )
    era_penalty = -abs(row["year"] - item["year"]) * 0.02
    unrounded = {
        "platform_match": platform_match, "genre_match": genre_match, "rating_match": rating_match,
        "publisher_match": publisher_match, "critic_closeness": critic_closeness, "era_penalty": era_penalty,
    }
    total = sum(unrounded.values())  # unrounded, matches _comparable_score's return exactly
    return {**{k: round(v, 3) for k, v in unrounded.items()}, "total": total}


def _comparable_score(row: dict, item: dict) -> float:
    """Weighted-match score for how similar a past release is to a query.

    Not a learned embedding: platform and genre matter most (a shooter's comps
    are other shooters), rating and publisher less, critic score only when
    both sides have one, with a small penalty for being from a different
    console generation.
    """
    return _comparable_score_breakdown(row, item)["total"]


def find_comparables(a: Artifacts, item: dict, k: int = 8) -> list[dict]:
    """The most similar past releases, and what actually happened to them.

    Only releases that actually share something with the query are returned
    -- an empty or short list is the honest answer when nothing comparable
    exists, not a reason to pad with unrelated titles.
    """
    scored = [(_comparable_score(row, item), row) for row in a.catalog]
    scored = [(s, row) for s, row in scored if s > 0]
    scored.sort(key=lambda t: -t[0])
    return [{**row, "similarity": round(s, 2)} for s, row in scored[:k]]


def market_saturation(a: Artifacts, genre: str, platform: str) -> dict:
    """Real historical saturation for one genre/platform combo -- no concept, no model call.

    Shared by validate_concept() (which adds a probability on top) and the AI
    market-analyst's get_market_saturation tool, so both read the same numbers.
    """
    same_combo = [row for row in a.catalog if row["platform"] == platform and row["genre"] == genre]
    recent_cutoff = HISTORY_THROUGH - 3
    recent_same_combo = [row for row in same_combo if row["year"] >= recent_cutoff]
    hit_rate = (sum(row["is_hit"] for row in same_combo) / len(same_combo)) if same_combo else None
    return {
        "same_genre_and_platform_releases": len(same_combo),
        "same_genre_and_platform_hit_rate": hit_rate,
        "same_genre_and_platform_recent_releases": len(recent_same_combo),
        "recent_window": f"{recent_cutoff}-{HISTORY_THROUGH}",
        "note": (f"Counts are historical releases in this dataset (through {HISTORY_THROUGH}); "
                 "they describe that record, not today's live market."),
    }


def validate_concept(a: Artifacts, item: dict, k: int = 8) -> dict:
    """Pre-launch concept check: real comparables, a probability, and honest market signals.

    Not a new model -- a synthesis of predict() and find_comparables() plus
    simple, fully-real aggregates over the catalog (no field here is invented
    or estimated beyond what the model already does for `prediction`).
    """
    comparables = find_comparables(a, item, k)
    saturation = market_saturation(a, item["genre"], item["platform"])
    same_combo_n = saturation["same_genre_and_platform_releases"]
    hit_rate = saturation["same_genre_and_platform_hit_rate"]

    risk_factors = []
    if not comparables:
        risk_factors.append(
            "No past release in this dataset shares platform, genre, rating, or publisher "
            "with this concept -- the prediction below has no real precedent behind it."
        )
    if hit_rate is not None and hit_rate < a.base_rate * 0.7:
        risk_factors.append(
            f"{item['genre']} on {item['platform']} hit (sold 1M+) {hit_rate:.0%} of the time in this "
            f"dataset, well below the {a.base_rate:.0%} rate across all releases."
        )
    if same_combo_n and not saturation["same_genre_and_platform_recent_releases"]:
        risk_factors.append(
            f"No {item['genre']}/{item['platform']} release in {saturation['recent_window']}, the most "
            "recent years this dataset covers -- the combination may be fading, or this is just a thin slice."
        )

    return {
        "prediction": predict(a, [item])[0] | {"base_rate": a.base_rate},
        "comparables": comparables,
        "market_saturation": saturation,
        "price_distribution": None,
        "price_distribution_note": "Not available: this dataset has no price field for historical releases.",
        "risk_factors": risk_factors,
    }


def load_metrics() -> dict:
    import json
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


if __name__ == "__main__":
    built = build_artifacts()
    save_artifacts(built)
    meta = write_meta(built)
    print(f"wrote {MODEL_PATH} and {META_PATH.name} (artifact sha256 {meta['artifact_sha256'][:16]}...)")
