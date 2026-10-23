"""Application entry point.

Run from the backend/ directory:  uvicorn app:app --reload
"""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.routes import error_response, router as api_router

load_dotenv()

SERVICE_NAME = "weather-route-optimizer"

# Used only when CORS_ORIGINS is not set: the local Vite dev server.
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

app = FastAPI(title="Weather-Aware Route Optimizer")

# Allowed frontend origins come from .env (CORS_ORIGINS, comma-separated).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if o.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return request validation errors in the API's common error shape."""
    details = [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
        for err in exc.errors()
    ]
    message = "; ".join(f"{d['field']}: {d['message']}" for d in details) or "Invalid request"
    return error_response(422, "invalid_request", message, details)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE_NAME}
