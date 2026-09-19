"""HTTP API and dashboard for the hit-prediction model.

Run: ``uvicorn app.main:app --reload`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import service

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Video Game Hit Predictor",
    description="Probability that a release sells 1M+ units, from launch-time features.",
    version="1.0.0",
)


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
