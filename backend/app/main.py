"""
main.py
========
FastAPI application entrypoint for the xNIDS demo project.

On startup we:
  1. Initialize the database (creates tables if absent).
  2. Train (or load cached) the two detectors: Kitsune-style AE
     (paper baseline) and LSTM-Autoencoder (our DL alternative).

Run with:  uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.database import init_db
from .core.model_store import store
from .routers import detect, attacks, defense, dashboard

app = FastAPI(
    title="xNIDS Demo Platform",
    description=("Full-stack demonstration platform implementing the xNIDS paper "
                  "(USENIX Security 2023) baseline + a custom LSTM-Autoencoder DL "
                  "alternative, with live attack simulation, XNIDS-style explanation, "
                  "and automated defense-rule generation."),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    store.train_or_load()


@app.get("/")
def root():
    return {"status": "ok", "service": "xNIDS Demo API", "models_ready": store.ready}


@app.get("/api/health")
def health():
    return {"status": "healthy", "models_ready": store.ready}


app.include_router(detect.router, prefix="/api", tags=["detection"])
app.include_router(attacks.router, prefix="/api", tags=["attacks"])
app.include_router(defense.router, prefix="/api", tags=["defense"])
app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
