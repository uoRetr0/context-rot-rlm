"""Configuration loader: merges config.yaml with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml() -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_YAML: dict[str, Any] = _load_yaml()


class Settings(BaseSettings):
    google_api_key: str = Field(default="")

    # Models
    model_fast: str = _YAML["models"]["fast"]
    model_pro: str = _YAML["models"]["pro"]
    model_embedding: str = _YAML["models"]["embedding"]

    # Budget
    max_dollars: float = _YAML["budget"]["max_dollars"]
    warn_at_dollars: float = _YAML["budget"]["warn_at_dollars"]

    # Chunking
    chunk_size: int = _YAML["chunking"]["chunk_size"]
    chunk_overlap: int = _YAML["chunking"]["chunk_overlap"]
    min_chunk_size: int = _YAML["chunking"]["min_chunk_size"]

    # Retrieval
    bm25_top_k: int = _YAML["retrieval"]["bm25_top_k"]
    vector_top_k: int = _YAML["retrieval"]["vector_top_k"]
    hybrid_top_k: int = _YAML["retrieval"]["hybrid_top_k"]
    rrf_k: int = _YAML["retrieval"]["rrf_k"]

    # RLM
    rlm_max_depth: int = _YAML["rlm"]["max_depth"]
    rlm_confidence_threshold: float = _YAML["rlm"]["confidence_threshold"]
    rlm_max_sub_questions: int = _YAML["rlm"]["max_sub_questions"]
    rlm_max_chunks_per_step: int = _YAML["rlm"]["max_chunks_per_step"]
    rlm_initial_chunks: int = _YAML["rlm"]["initial_chunks"]
    rlm_sub_question_chunks: int = _YAML["rlm"]["sub_question_chunks"]

    # Baselines
    fullcontext_max_tokens: int = _YAML["fullcontext"]["max_input_tokens"]
    rag_top_k: int = _YAML["rag"]["top_k"]
    mapreduce_map_chunks: int = _YAML["mapreduce"]["map_chunk_count"]
    mapreduce_reduce_max: int = _YAML["mapreduce"]["reduce_max_tokens"]

    # Benchmarks
    benchmark_cfg: dict[str, Any] = _YAML["benchmarks"]

    # Experiment
    seed: int = _YAML["experiment"]["seed"]
    methods: list[str] = _YAML["experiment"]["methods"]
    output_dir: str = _YAML["experiment"]["output_dir"]

    model_config = {"env_prefix": "", "extra": "ignore"}


settings = Settings()
