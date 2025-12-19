"""
RAG Configuration and Search Space Definition.

This module defines the hyperparameter search space for RAG optimization,
based on academic literature and best practices.

References:
- Barker et al. (2025) - "Faster, Cheaper, Better: Multi-Objective HPO for RAG"
- MTEB Benchmark for embedding model selection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import itertools


# =============================================================================
# EMBEDDING MODELS - Based on MTEB Benchmark Rankings
# =============================================================================
EMBEDDING_MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "speed": "fast",
        "mteb_score": 56.0,
    },
    "mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "speed": "medium",
        "mteb_score": 57.8,
    },
    "bge": {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "speed": "medium",
        "mteb_score": 63.5,
    },
}

# =============================================================================
# GENERATOR MODELS - For LLM-based answer generation
# =============================================================================
GENERATOR_MODELS = {
    "flan-t5-small": {
        "name": "google/flan-t5-small",
        "params": "80M",
        "speed": "fast",
    },
    "flan-t5-base": {
        "name": "google/flan-t5-base",
        "params": "250M",
        "speed": "medium",
    },
}

# =============================================================================
# SEARCH SPACE DEFINITION
# =============================================================================
@dataclass
class RAGSearchSpace:
    """
    Complete RAG hyperparameter search space.
    
    Based on:
    - Barker et al. (2025): 50 iterations for RAG HPO
    - Standard RAG pipeline parameters from literature
    """
    
    # Chunking parameters
    # chunk_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 896, 1024])
    chunk_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    chunk_overlaps: List[int] = field(default_factory=lambda: [0, 16, 32, 64,])
    
    # Retrieval parameters
    top_k_values: List[int] = field(default_factory=lambda: list(range(3, 8)))
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    retrieval_metrics: List[str] = field(default_factory=lambda: ["cosine", "dot"])
    
    # Model selection
    # embedding_models: List[str] = field(default_factory=lambda: ["minilm", "mpnet", "bge"])
    embedding_models: List[str] = field(default_factory=lambda: ["minilm"])

    # Context window
    context_windows: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    
    def get_param_dict(self) -> Dict[str, List[Any]]:
        """Return search space as dictionary for algorithms."""
        return {
            "chunk_size": self.chunk_sizes,
            "chunk_overlap": self.chunk_overlaps,
            "top_k": self.top_k_values,
            "similarity_threshold": self.similarity_thresholds,
            "retrieval_metric": self.retrieval_metrics,
            "embedding_model": self.embedding_models,
            "max_context_chars": self.context_windows,
        }
    
    def get_total_configurations(self) -> int:
        """Calculate total number of possible configurations."""
        return (
            len(self.chunk_sizes) *
            len(self.chunk_overlaps) *
            len(self.top_k_values) *
            len(self.similarity_thresholds) *
            len(self.retrieval_metrics) *
            len(self.embedding_models) *
            len(self.context_windows)
        )
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get min/max ranges for each parameter."""
        return {
            "chunk_size": (min(self.chunk_sizes), max(self.chunk_sizes)),
            "chunk_overlap": (min(self.chunk_overlaps), max(self.chunk_overlaps)),
            "top_k": (min(self.top_k_values), max(self.top_k_values)),
            "similarity_threshold": (min(self.similarity_thresholds), max(self.similarity_thresholds)),
            "max_context_chars": (min(self.context_windows), max(self.context_windows)),
        }


@dataclass
class RAGConfig:
    """
    A single RAG configuration (one point in the search space).
    """
    chunk_size: int = 256
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.2
    retrieval_metric: str = "cosine"
    embedding_model: str = "minilm"
    
    # Generator settings (fixed for fair comparison)
    generator_model: str = "flan-t5-small"
    max_new_tokens: int = 64
    temperature: float = 0.0
    max_context_chars: int = 2048
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_metric": self.retrieval_metric,
            "embedding_model": self.embedding_model,
            "generator_model": self.generator_model,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "max_context_chars": self.max_context_chars,
        }
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for caching."""
        return (
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
            self.similarity_threshold,
            self.retrieval_metric,
            self.embedding_model,
            self.generator_model,
            self.max_new_tokens,
            self.temperature,
            self.max_context_chars,
        )
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RAGConfig":
        """Create from dictionary."""
        chunk_size = int(d.get("chunk_size", 256))
        # Default overlap scales with chunk size; clamp to a valid range.
        default_overlap = min(64, max(0, chunk_size // 4))
        chunk_overlap = int(d.get("chunk_overlap", default_overlap))
        if chunk_size > 0:
            chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=d.get("top_k", 5),
            similarity_threshold=d.get("similarity_threshold", 0.2),
            retrieval_metric=d.get("retrieval_metric", "cosine"),
            embedding_model=d.get("embedding_model", "minilm"),
            generator_model=d.get("generator_model", "flan-t5-small"),
            max_new_tokens=d.get("max_new_tokens", 64),
            temperature=d.get("temperature", 0.0),
            max_context_chars=d.get("max_context_chars", 2048),
        )
    
    def get_embedding_model_name(self) -> str:
        """Get full embedding model name from short key."""
        return EMBEDDING_MODELS.get(self.embedding_model, EMBEDDING_MODELS["minilm"])["name"]
    
    def get_generator_model_name(self) -> str:
        """Get full generator model name from short key."""
        return GENERATOR_MODELS.get(self.generator_model, GENERATOR_MODELS["flan-t5-small"])["name"]


# =============================================================================
# EXPERIMENT SETTINGS - Based on Literature
# =============================================================================
@dataclass
class ExperimentSettings:
    """
    Experiment configuration based on academic best practices.
    
    Reference: Barker et al. (2025) uses 50 total iterations for RAG HPO.
    """
    # Evaluation budget per algorithm run
    max_evaluations: int = 50
    
    # Number of independent runs for statistical significance
    num_runs: int = 5
    
    # Dataset settings
    dataset_size: int = 500  # Total QA pairs to load
    eval_sample_size: int = 50  # QA pairs per fitness evaluation
    dataset_seed: int = -1  # -1 = draw a random seed each run

    # Algorithm-specific settings
    hill_climbing_restarts: int = 5
    sa_initial_temperature: float = 0.5
    sa_cooling_rate: float = 0.80
    sa_min_temperature: float = 0.001
    
    # Parallel processing
    num_workers: int = 8


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================
DEFAULT_SEARCH_SPACE = RAGSearchSpace()
DEFAULT_SETTINGS = ExperimentSettings()

# Simplified search space for faster experiments
# Note: The CLI runner (`src/main.py`) uses `src/rag/search_space.py` and its
# `DEFAULT_SEARCH_SPACE` by default. These "FAST_*" presets are provided as
# optional utilities for quick local experiments and are not automatically used.
FAST_SEARCH_SPACE = RAGSearchSpace(
    chunk_sizes=[256, 512, 768],
    chunk_overlaps=[0, 48],
    top_k_values=[3, 5, 7],
    similarity_thresholds=[0.2, 0.4],
    retrieval_metrics=["cosine"],
    embedding_models=["minilm"],
    context_windows=[1024, 2048],
)

FAST_SETTINGS = ExperimentSettings(
    max_evaluations=30,
    num_runs=5,
    dataset_size=200,
    eval_sample_size=50,
)
