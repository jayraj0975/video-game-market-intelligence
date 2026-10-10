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


def test_comparables_returns_real_similar_releases():
    r = client.post("/api/comparables", json=GOOD)
    assert r.status_code == 200
    rows = r.json()["comparables"]
    assert rows, "a well-known PS3 shooter from a major publisher should have comparables"
    assert all(row["platform"] == "PS3" or row["genre"] == "Shooter" for row in rows)
    assert rows == sorted(rows, key=lambda row: -row["similarity"])
    assert "is_hit" in rows[0] and "global_sales_munits" in rows[0]


def test_comparables_do_not_pad_with_unrelated_titles():
    from app import service
    catalog = [{"title": "Totally Unrelated", "platform": "PC", "genre": "Puzzle", "rating": "E",
                "publisher": "Someone Else", "year": 1998, "critic_score": None,
                "global_sales_munits": 0.1, "is_hit": 0}]
    a = service.get_artifacts()
    stub = service.Artifacts(**{**a.__dict__, "catalog": catalog})
    out = service.find_comparables(stub, {
        "platform": "PS3", "genre": "Shooter", "rating": "M", "publisher": "Activision",
        "year": 2013, "critic_score": 75})
    assert out == []  # nothing in the catalog shares platform, genre, rating or publisher


def test_concept_validate_returns_prediction_comparables_and_saturation():
    r = client.post("/api/concept-validate", json={
        "platform": "PS3", "genre": "Shooter", "rating": "M", "year": 2015})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prediction"]["probability"] <= 1
    assert body["comparables"], "a PS3 shooter should have real historical comparables"
    sat = body["market_saturation"]
    assert sat["same_genre_and_platform_releases"] > 0
    assert sat["same_genre_and_platform_hit_rate"] is not None
    assert body["price_distribution"] is None
    assert "not available" in body["price_distribution_note"].lower()
    assert isinstance(body["risk_factors"], list)


def test_concept_validate_flags_no_precedent_honestly():
    from app import service
    catalog = [{"title": "Unrelated", "platform": "PC", "genre": "Puzzle", "rating": "E",
                "publisher": "Someone Else", "year": 1998, "critic_score": None,
                "global_sales_munits": 0.1, "is_hit": 0}]
    a = service.get_artifacts()
    stub = service.Artifacts(**{**a.__dict__, "catalog": catalog})
    out = service.validate_concept(stub, {
        "title": "", "platform": "PS3", "genre": "Shooter", "rating": "M", "publisher": "",
        "year": 2015, "critic_score": None, "critic_count": None, "n_platforms": 1})
    assert out["comparables"] == []
    assert out["market_saturation"]["same_genre_and_platform_releases"] == 0
    assert out["market_saturation"]["same_genre_and_platform_hit_rate"] is None
    assert any("no past release" in f.lower() for f in out["risk_factors"])


def test_metrics_and_market_and_index():
    assert "models" in client.get("/api/metrics").json()
    assert client.get("/api/market").json()["hit_rate_by_score"]
    page = client.get("/")
    assert page.status_code == 200
    # Strict policy: the page's script is a file, not inline, and nothing loads from other sites.
    assert "script-src 'self';" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert '<script src="/static/app.js">' in page.text and client.get("/static/app.js").status_code == 200


def test_sensitivity_rescores_one_change_at_a_time():
    release = {"title": "Kart Party", "platform": "Wii", "genre": "Racing", "rating": "E", "publisher": "Nintendo", "year": 2009,
               "critic_score": 80, "critic_count": 40}
    r = client.post("/api/sensitivity", json=release)
    assert r.status_code == 200
    s = r.json()
    base = client.post("/api/predict", json=release).json()["probability"]
    assert s["base_probability"] == pytest.approx(base)
    assert set(s["dimensions"]) == {"platform", "genre", "rating", "n_platforms"}
    for dim, d in s["dimensions"].items():
        probs = [o["probability"] for o in d["options"] if not o["current"]]
        assert probs == sorted(probs, reverse=True), dim  # best alternatives first
        current = [o for o in d["options"] if o["current"]]
        assert len(current) == 1 and current[0]["probability"] == pytest.approx(base), dim  # where it is now is always shown
        assert all(o["ratio"] == pytest.approx(o["probability"] / base) for o in d["options"])
    assert "not what the change would cause" in s["note"]


def test_sensitivity_offers_only_alternatives_with_history():
    s = client.post("/api/sensitivity", json={"title": "Space Raiders", "platform": "PS4", "genre": "Shooter", "rating": "M",
                                              "publisher": "Ubisoft", "year": 2015}).json()["dimensions"]
    ratings = {o["value"] for o in s["rating"]["options"]}
    assert not ratings & {"AO", "EC", "K-A", "RP", "Unrated"}  # a handful of titles each, or no rating at all
    platforms = {o["value"] for o in s["platform"]["options"]}
    assert not platforms & {"PS", "N64", "GBA", "DC"}  # long gone by 2015
    assert s["platform"]["considered"] < 15


def test_sensitivity_rejects_bad_input():
    assert client.post("/api/sensitivity", json={"title": "x", "platform": "Nope", "genre": "Racing", "rating": "E",
                                                 "publisher": "", "year": 2009}).status_code == 422
