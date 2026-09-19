import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD = {"title": "Call of Duty: Ghosts", "platform": "PS3", "genre": "Shooter",
        "rating": "M", "publisher": "Activision", "year": 2013,
        "critic_score": 75, "critic_count": 60, "n_platforms": 4}


def test_options_lists_choices():
    r = client.get("/api/options").json()
    assert "PS3" in r["platforms"] and "Shooter" in r["genres"]
    assert 0 < r["base_rate"] < 1


def test_predict_returns_a_probability():
    r = client.post("/api/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["franchise_history"]["titles"] > 0


def test_established_franchise_beats_unknown_obscure_title():
    strong = client.post("/api/predict", json=GOOD).json()["probability"]
    weak = client.post("/api/predict", json={
        **GOOD, "title": "Zzyzx Adventure", "publisher": "Nobody Games",
        "critic_score": None, "critic_count": None, "n_platforms": 1}).json()["probability"]
    assert strong > weak


def test_rank_orders_by_probability():
    weak = {**GOOD, "title": "Zzyzx Adventure", "publisher": "Nobody Games",
            "critic_score": None, "critic_count": None, "n_platforms": 1}
    r = client.post("/api/rank", json={"releases": [weak, GOOD]}).json()["ranked"]
    assert [x["rank"] for x in r] == [1, 2]
    assert r[0]["probability"] >= r[1]["probability"]
    assert r[0]["title"] == GOOD["title"]


@pytest.mark.parametrize("patch", [
    {"platform": "Atari 9000"}, {"genre": "Nope"}, {"critic_score": 150},
    {"critic_count": None},  # score without count
])
def test_bad_input_is_rejected(patch):
    assert client.post("/api/predict", json={**GOOD, **patch}).status_code == 422


def test_metrics_and_market_and_index():
    assert "models" in client.get("/api/metrics").json()
    assert client.get("/api/market").json()["hit_rate_by_score"]
    assert client.get("/").status_code == 200
