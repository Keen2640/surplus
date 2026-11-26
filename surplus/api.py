"""FastAPI service + dashboard.   uvicorn surplus.api:app --port 8040"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .curve import bootstrap
from .engine import Engine

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Surplus: pension LDI engine")
engine = Engine()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/api/summary")
def summary():
    return engine.liability_summary()


@app.get("/api/curve")
def curve():
    p, c = engine.par, engine.curve
    b = bootstrap(p)
    t = np.round(np.linspace(0.25, 30, 120), 3)
    return {"date": p.date, "par": {"tenor": p.tenors.tolist(), "yield": (p.yields * 100).tolist()},
            "t": t.tolist(), "nss_zero": (c.zero(t) * 100).tolist(), "boot_zero": (b.zero(t) * 100).tolist(),
            "fwd": (c.fwd(t) * 100).tolist(), "fit": engine.curve_fit, "params": c.meta["params"]}


@app.get("/api/cashflows")
def cashflows():
    t, a = engine.cashflows
    keep = t <= 80
    return {"t": t[keep].tolist(), "amount": a[keep].tolist(), "census": engine.census.summary()}


@app.get("/api/hedge")
def hedge(target: float = Query(0.9, gt=0, le=1.5), physical: float = Query(0.4, gt=0, le=1), swaps: bool = True):
    return engine.hedge(target, physical, swaps).as_dict()


@app.get("/api/shock")
def shock(bp: float = Query(-100, ge=-500, le=500), tenor: float | None = None):
    return engine.rate_shock(bp, tenor)


@app.get("/api/longevity")
def longevity(delta: float = Query(0.005, ge=-0.02, le=0.03)):
    return engine.longevity(delta)


@lru_cache(maxsize=16)
def _project(years: int, n: int, rates_only: bool):
    res = engine.project_all(years, n, equity_vol=0.0 if rates_only else 0.16)
    return {"strategies": [{"metrics": r.metrics(), "fan": r.fan()} for r in res]}


@app.get("/api/project")
def project(years: int = Query(10, ge=1, le=30), n: int = Query(5000, ge=500, le=20000), rates_only: bool = False):
    return _project(years, n, rates_only)


class Question(BaseModel):
    question: str


@app.post("/api/ask")
def ask_analyst(q: Question):
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or (Path.home() / ".config" / "anthropic").exists()):
        raise HTTPException(503, "The AI analyst needs Claude API credentials (set ANTHROPIC_API_KEY).")
    from .analyst import ask

    a = ask(q.question[:2000], engine)
    return {"answer": a.text, "tool_calls": [{"tool": c["tool"], "input": c["input"]} for c in a.tool_calls],
            "audit": a.audit}
