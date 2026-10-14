"""
FastAPI server for the Edge-AI Compression Gym.

The dev branch exposes the final selected environment configuration with the
single supported reward function, `hybrid_a`.
"""

from __future__ import annotations

import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVER_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from openenv.core.env_server import create_fastapi_app
from models import CompressionAction, CompressionObservation
from server.environment import CompressionEnvironment


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = create_fastapi_app(
        CompressionEnvironment,
        CompressionAction,
        CompressionObservation,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "environment": "CompressionEnvironment",
            "tasks": ["easy", "medium", "hard"],
            "selected_reward_function": "hybrid_a",
            "recommended_quantization_bits": [16, 8, 16],
            "recommended_prune_sparsities": [0.25, 0.35],
        }

    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Console entrypoint for `uv run server` and openenv multi-mode validation."""
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
