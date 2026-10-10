"""HTTP API and dashboard for the hit-prediction model.

Run: ``uvicorn app.main:app --reload`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import ai, service

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DECISION_POINT_LABEL, DECISION_POINT_NOTE, MODEL_VERSION  # noqa: E402

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Video Game Hit Predictor",
    description=(
        "Probability that a release sells 1M+ units, from features known shortly before release. "
        f"Decision point: {DECISION_POINT_LABEL}. {DECISION_POINT_NOTE}"
    ),
    version=MODEL_VERSION,
)


# Same-origin only: the page, its script and its API all live here. The docs pages load Swagger UI from a CDN, so
# they keep the browser defaults.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path not in ("/docs", "/redoc"):
        response.headers["Content-Security-Policy"] = CSP
    return response


class Release(BaseModel):
    title: str = Field("", max_length=120, description="Used to find the franchise's track record")
    platform: str
    genre: str
    rating: str
    publisher: str = Field("", max_length=120)
    year: int = Field(2015, ge=1996, le=2030)
    critic_score: float | None = Field(None, ge=0, le=100)
    critic_count: int | None = Field(None, ge=0, le=200)
    n_platforms: int = Field(1, ge=1, le=12)


class Slate(BaseModel):
    releases: list[Release] = Field(..., min_length=1, max_length=50)


class MarketQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


def _validated(releases: list[Release]) -> list[dict]:
    a = service.get_artifacts()
    for i, r in enumerate(releases):
        for value, allowed, name in ((r.platform, a.platforms, "platform"),
                                     (r.genre, a.genres, "genre"),
                                     (r.rating, a.ratings, "rating")):
            if value not in allowed:
                raise HTTPException(422, f"release {i}: unknown {name} '{value}'")
        if (r.critic_score is None) != (r.critic_count is None):
            raise HTTPException(422, f"release {i}: give both critic_score and critic_count, or neither")
    return [r.model_dump() for r in releases]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict:
    a = service.get_artifacts()
    top = sorted(a.publishers, key=lambda k: -a.publishers[k][0])
    return {"platforms": a.platforms, "genres": a.genres, "ratings": a.ratings,
            "publishers": top, "base_rate": a.base_rate}


@app.post("/api/predict")
def predict(release: Release) -> dict:
    a = service.get_artifacts()
    return service.predict(a, _validated([release]))[0] | {"base_rate": a.base_rate}


@app.post("/api/rank")
def rank(slate: Slate) -> dict:
    a = service.get_artifacts()
    scored = service.predict(a, _validated(slate.releases))
    ranked = sorted(scored, key=lambda r: -r["probability"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {"ranked": ranked, "base_rate": a.base_rate}


@app.post("/api/comparables")
def comparables(release: Release) -> dict:
    """The most similar past releases and what actually happened to them."""
    a = service.get_artifacts()
    item = _validated([release])[0]
    return {"comparables": service.find_comparables(a, item)}


@app.post("/api/concept-validate")
def concept_validate(release: Release) -> dict:
    """Should I make this game? Real comparables, a probability, and honest market signals
    for a concept that doesn't exist yet -- leave title/publisher/critic fields blank."""
    a = service.get_artifacts()
    item = _validated([release])[0]
    return service.validate_concept(a, item)


@app.post("/api/market-analysis")
def market_analysis(q: MarketQuestion) -> dict:
    """Ask the AI market analyst a natural-language question. Investigates via real tool
    calls over find_comparables/market_saturation; comparable and saturation figures in the
    response always come from those real calls, never from the model's own restated text.
    Returns {"available": false, ...} rather than a fake answer if no API key is configured."""
    return ai.analyze_market(q.question)


@app.get("/api/model-info")
def model_info() -> dict:
    """What is being served and where it came from: version, commit, data hashes, windows, artifact hash."""
    meta = service.load_meta()
    if meta is None:
        raise HTTPException(503, "model provenance file is missing; rebuild with `python -m app.service`")
    return meta


@app.get("/api/metrics")
def metrics() -> dict:
    return service.load_metrics()


@app.get("/api/market")
def market() -> dict:
    return service.get_artifacts().market


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
