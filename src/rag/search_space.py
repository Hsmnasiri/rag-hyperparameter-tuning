"""
RAG Hyperparameter Search Space Configuration.

Based on academic literature:
- AutoRAG-HP (Fu et al., EMNLP 2024)
- Barker et al. (ICLR 2025)
- RAGBench (Belyi et al., 2024)

This module defines the complete search space for RAG hyperparameter optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import itertools


@dataclass
class SearchSpace:
    """
    Defines the hyperparameter search space for RAG optimization.
    
    Parameters are chosen based on academic benchmarks and best practices.
    """
    
    # Chunking parameters (include tiny/huge to cover bad/good configs)
    chunk_sizes: List[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
    )
    chunk_overlaps: List[int] = field(
        default_factory=lambda: [0, 16, 32, 48, 64, 96, 128]
    )
    
    # Retrieval parameters (include weak settings)
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 11, 13, 17, 19])
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8])
    
    # Embedding models (keep lightweight by default)
    embedding_models: List[str] = field(default_factory=lambda: ["minilm"])
    
    # Retrieval metrics
    retrieval_metrics: List[str] = field(default_factory=lambda: ["cosine", "dot"])

    # Context window limits (characters)
    context_windows: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    
    def get_config_space(self) -> Dict[str, List[Any]]:
        """Return the complete configuration space as a dictionary."""
        return {
            "chunk_size": self.chunk_sizes,
            "chunk_overlap": self.chunk_overlaps,
            "top_k": self.top_k_values,
            "similarity_threshold": self.similarity_thresholds,
            "embedding_model": self.embedding_models,
            "retrieval_metric": self.retrieval_metrics,
            "max_context_chars": self.context_windows,
        }
    
    def get_total_configurations(self) -> int:
        """Calculate total number of possible configurations."""
        space = self.get_config_space()
        total = 1
        for values in space.values():
            total *= len(values)
        return total
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        import random
        space = self.get_config_space()
        config = {key: random.choice(values) for key, values in space.items()}
        return config

    @classmethod
    def from_dict(cls, space: Dict[str, List[Any]]) -> "SearchSpace":
        """
        Build a SearchSpace instance from a plain dictionary.
        Missing keys fall back to class defaults.
        """
        defaults = cls()
        return cls(
            chunk_sizes=space.get("chunk_size", defaults.chunk_sizes),
            chunk_overlaps=space.get("chunk_overlap", defaults.chunk_overlaps),
            top_k_values=space.get("top_k", defaults.top_k_values),
            similarity_thresholds=space.get("similarity_threshold", defaults.similarity_thresholds),
            embedding_models=space.get("embedding_model", defaults.embedding_models),
            retrieval_metrics=space.get("retrieval_metric", defaults.retrieval_metrics),
            context_windows=space.get("max_context_chars", defaults.context_windows),
        )
    
    def get_neighbors(
        self, 
        config: Dict[str, Any],
        include_diagonal: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring configurations (for local search algorithms).
        
        A neighbor differs in exactly one parameter value (or two if diagonal).
        """
        space = self.get_config_space()
        neighbors: List[Dict[str, Any]] = []
        
        for param, values in space.items():
            if param not in config:
                continue
            current_value = config[param]
            if current_value not in values:
                continue
            
            current_idx = values.index(current_value)
            
            # Try adjacent values
            for delta in [-1, 1]:
                new_idx = current_idx + delta
                if 0 <= new_idx < len(values):
                    neighbor = config.copy()
                    neighbor[param] = values[new_idx]
                    if neighbor != config and neighbor not in neighbors:
                        neighbors.append(neighbor)
        
        if include_diagonal:
            # Add some diagonal moves (change two parameters at once)
            params = list(space.keys())
            for i, param1 in enumerate(params):
                for param2 in params[i+1:]:
                    values1 = space[param1]
                    values2 = space[param2]
                    
                    idx1 = values1.index(config[param1]) if config.get(param1) in values1 else 0
                    idx2 = values2.index(config[param2]) if config.get(param2) in values2 else 0
                    
                    for d1, d2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        new_idx1 = idx1 + d1
                        new_idx2 = idx2 + d2
                        
                        if 0 <= new_idx1 < len(values1) and 0 <= new_idx2 < len(values2):
                            neighbor = config.copy()
                            neighbor[param1] = values1[new_idx1]
                            neighbor[param2] = values2[new_idx2]
                            if neighbor != config and neighbor not in neighbors:
                                neighbors.append(neighbor)
        
        return neighbors
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a configuration against the search space.
        
        Returns:
            (is_valid, error_message)
        """
        space = self.get_config_space()
        
        for param, values in space.items():
            if param not in config:
                return False, f"Missing parameter: {param}"
            if config[param] not in values:
                return False, f"Invalid value for {param}: {config[param]}"
        
        # Additional validation rules
        if config["chunk_overlap"] >= config["chunk_size"]:
            return False, "chunk_overlap must be less than chunk_size"
        if config.get("max_context_chars", 0) <= 0:
            return False, "max_context_chars must be positive"
        
        return True, None
    
    def __str__(self) -> str:
        space = self.get_config_space()
        lines = ["RAG Search Space:"]
        for param, values in space.items():
            lines.append(f"  {param}: {values}")
        lines.append(f"  Total configurations: {self.get_total_configurations():,}")
        return "\n".join(lines)


# Default search space instance
DEFAULT_SEARCH_SPACE = SearchSpace()


# Simplified search space for faster experiments
FAST_SEARCH_SPACE = SearchSpace(
    chunk_sizes=[256, 512, 768],
    chunk_overlaps=[0, 32],
    top_k_values=[3, 5, 7],
    similarity_thresholds=[0.2, 0.4],
    embedding_models=["minilm"],
    retrieval_metrics=["cosine"],
    context_windows=[1024, 2048],
)


@dataclass
class RAGConfig:
    """
    A single RAG configuration with all hyperparameters.
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_metric: str = "cosine"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model,
            "retrieval_metric": self.retrieval_metric,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RAGConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def __hash__(self):
        return hash((
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
            self.similarity_threshold,
            self.embedding_model,
            self.retrieval_metric,
        ))
    
    def __eq__(self, other):
        if not isinstance(other, RAGConfig):
            return False
        return self.to_dict() == other.to_dict()


def config_to_tuple(config: Dict[str, Any]) -> tuple:
    """Convert config dict to hashable tuple for caching."""
    return (
        config.get("chunk_size", 512),
        config.get("chunk_overlap", 64),
        config.get("top_k", 5),
        config.get("similarity_threshold", 0.0),
        config.get("embedding_model", "minilm"),
        config.get("retrieval_metric", "cosine"),
        config.get("max_context_chars", 2048),
    )
