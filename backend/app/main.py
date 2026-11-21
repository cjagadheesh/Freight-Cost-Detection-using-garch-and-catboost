"""FastAPI application. Run from ``backend/``:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.state import AppState
from app.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState(settings)
    app.state.app_state = state
    try:
        state.load()
    except Exception as exc:  # start anyway so /health and /status can report it
        logger.exception("Startup load failed")
        state.last_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description="Dry-bulk freight forecasting and chartering decisions on real market data.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
def health(request: Request):
    st: AppState = request.app.state.app_state
    body = {"status": "ok" if st.ready else "degraded", "ready": st.ready}
    return JSONResponse(body, status_code=200)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)
