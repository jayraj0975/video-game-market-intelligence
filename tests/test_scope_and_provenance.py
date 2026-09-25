"""What the model is allowed to know, what it says it is, and where it came from. Offline: uses only the
committed model and its provenance file."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import service
from app.main import Release, app
from config import DECISION_POINT_LABEL, MODEL_MAX_YEAR, MODEL_VERSION
from features import CATEGORICAL, FEATURES, NUMERIC, POST_RELEASE

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)

# The complete list of things the model may look at. Adding a column means editing this set on
# purpose, which is the moment to ask whether it exists before the game ships.
KNOWN_BEFORE_RELEASE = {
    "Platform", "Genre", "Rating", "Year_of_Release", "platform_age", "is_sequel",
    "Critic_Score", "Critic_Count", "has_critic_score", "n_platforms_at_launch",
    "publisher_prior_titles", "publisher_prior_hits", "publisher_prior_hit_rate",
    "franchise_prior_titles", "franchise_prior_hits",
}
POST_RELEASE_HINTS = re.compile(r"sales|user_|users|global|units|revenue|review_count_after", re.I)


def test_features_are_exactly_the_known_before_release_set():
    assert set(FEATURES) == KNOWN_BEFORE_RELEASE
    assert set(FEATURES) == set(CATEGORICAL) | set(NUMERIC)
    assert len(FEATURES) == len(set(FEATURES))


def test_no_post_release_or_sales_columns_can_enter_the_model():
    assert not set(FEATURES) & set(POST_RELEASE)
    assert [f for f in FEATURES if POST_RELEASE_HINTS.search(f)] == []
    assert list(service.feature_frame(service.get_artifacts(), [
        {"title": "T", "platform": "PS3", "genre": "Action", "rating": "T", "publisher": "", "year": 2016}
    ]).columns) == FEATURES


def test_the_request_schema_does_not_accept_post_release_inputs():
    fields = set(Release.model_fields)
    assert not [f for f in fields if POST_RELEASE_HINTS.search(f)]
    assert {"critic_score", "critic_count"} <= fields  # near-launch inputs, and the API says so


def test_provenance_file_matches_the_committed_artifact():
    meta = service.load_meta()
    assert meta is not None, "run `python -m app.service` to write serving_model.meta.json"
    assert meta["artifact_sha256"] == hashlib.sha256(service.MODEL_PATH.read_bytes()).hexdigest()
    assert meta["artifact_bytes"] == service.MODEL_PATH.stat().st_size
    assert meta["features"] == FEATURES
    assert meta["feature_schema_sha256"] == hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
    assert meta["model_version"] == MODEL_VERSION
    assert meta["decision_point"] == DECISION_POINT_LABEL
    assert meta["history_through"] == MODEL_MAX_YEAR
    assert meta["training_window"] == "1996-2013" and meta["evaluation_window"].startswith("2014-2015")
    for key in ("code_commit", "scikit_learn", "python", "created_utc"):
        assert meta[key]
    assert set(meta["data_sha256"]) == {"raw_csv", "clean_csv"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in meta["data_sha256"].values())


def test_model_info_endpoint_serves_the_provenance():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["artifact_sha256"] == service.load_meta()["artifact_sha256"]
    assert "pre-release" in body["decision_point"].lower()
    assert "not a greenlight-stage model" in body["decision_point_note"]


GOOD = {"title": "Some New Game", "platform": "PS4", "genre": "Action", "rating": "T",
        "publisher": "Activision", "year": 2017}


def test_predictions_state_their_decision_point_and_critic_use():
    with_critics = client.post("/api/predict", json={**GOOD, "critic_score": 80, "critic_count": 40}).json()
    without = client.post("/api/predict", json=GOOD).json()
    assert with_critics["used_critic_data"] is True and without["used_critic_data"] is False
    assert "pre-release" in with_critics["decision_point"].lower()


def test_scoring_a_year_the_history_already_covers_is_flagged_as_not_a_backtest():
    old = client.post("/api/predict", json={**GOOD, "year": 2010}).json()
    new = client.post("/api/predict", json=GOOD).json()
    assert old["history_note"] and "not a valid historical backtest" in old["history_note"]
    assert new["history_note"] is None
    assert old["history_through"] == MODEL_MAX_YEAR


@pytest.mark.parametrize("path", ["README.md", "app/static/index.html"])
def test_docs_do_not_call_the_model_a_greenlight_model(path):
    text = (ROOT / path).read_text()
    assert not re.search(r"(knows?|knowable|known) at greenlight|greenlight-stage (estimate|model) [^.]*\bthe model\b", text, re.I), path
    assert "pre-release" in text.lower()
