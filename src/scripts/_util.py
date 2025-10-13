"""Utilities for script helpers.

This module provides small, script-local helpers shared by multiple scripts.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


def to_pgvector_literal(vec: np.ndarray) -> str:
    """Format a numpy vector as a pgvector literal, e.g. "[0.1,0.2,...]".

    Values are cast to float and formatted with 7 decimal places to keep payloads concise
    while remaining stable for indexing and comparison.
    """
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"

# Load .env once on import so all helpers see environment variables
load_dotenv()


def get_ollama_config() -> dict:
    """Read and validate Ollama configuration from environment.

    Required variables:
      - OLLAMA_MODEL (str)
      - RESPONSE_TEMPERATURE (float)
      - NUM_CTX (int)
      - NUM_PREDICT (int)

    Raises RuntimeError on any missing or invalid value.
    """
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise RuntimeError("OLLAMA_MODEL missing in .env")
    temp_str = os.getenv("RESPONSE_TEMPERATURE")
    if temp_str is None:
        raise RuntimeError("RESPONSE_TEMPERATURE missing in .env")
    ctx_str = os.getenv("NUM_CTX")
    if ctx_str is None:
        raise RuntimeError("NUM_CTX missing in .env")
    pred_str = os.getenv("NUM_PREDICT")
    if pred_str is None:
        raise RuntimeError("NUM_PREDICT missing in .env")

    try:
        temperature = float(temp_str)
    except ValueError as e:
        raise RuntimeError("RESPONSE_TEMPERATURE must be a float") from e
    try:
        num_ctx = int(ctx_str)
    except ValueError as e:
        raise RuntimeError("NUM_CTX must be an integer") from e
    try:
        num_predict = int(pred_str)
    except ValueError as e:
        raise RuntimeError("NUM_PREDICT must be an integer") from e

    return {
        "model": model,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }


def get_engine():
    """Create a SQLAlchemy engine using DATABASE_URL from the environment.

    Raises a RuntimeError if DATABASE_URL is missing.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return create_engine(db_url, future=True, pool_pre_ping=True)


def get_sentence_model() -> SentenceTransformer:
    """Load SentenceTransformer model from EMBED_MODEL env.

    Raises RuntimeError if EMBED_MODEL is missing.
    """
    model_name = os.getenv("EMBED_MODEL")
    if not model_name:
        raise RuntimeError("EMBED_MODEL missing in .env")
    return SentenceTransformer(model_name)


def init_engine_table_dim(exit_on_error: bool = True) -> tuple[Engine, str | None, int]:
    """Initialize and return (engine, table, embed_dim) from environment.

    If exit_on_error is True (default), print a concise message to stderr and exit(2)
    when required variables are missing; otherwise raise RuntimeError.
    Table may be None if not set; callers decide how to handle.
    """
    try:
        engine = get_engine()
        table = os.getenv("TABLE")
        embed_dim_str = os.getenv("EMBED_DIM")
        if embed_dim_str is None:
            raise RuntimeError("EMBED_DIM missing in .env")
        return engine, table, int(embed_dim_str)
    except RuntimeError as e:
        if exit_on_error:
            print(str(e), file=sys.stderr)
            raise SystemExit(2) from e
        raise
