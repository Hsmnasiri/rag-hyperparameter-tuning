"""
RAG Evaluation Module with LLM-based Generation.

Fitness Function (based on academic literature):
    fitness = 0.50 * F1_lexical      (PRIMARY signal)
            + 0.30 * semantic_sim    (Secondary signal)
            + 0.10 * exact_match     (Small bonus)
            + 0.10 * substring_match (Small bonus)

Key Features:
1. LLM-based answer generation as primary evaluator
2. Extractive QA as baseline for comparison
3. Complete configuration caching by full config tuple
4. Parallel evaluation support
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RAGConfig, ExperimentSettings, DEFAULT_SETTINGS, EMBEDDING_MODELS
from .pipeline import (
    RAGPipeline,
    DenseRetriever,
    LLMGenerator,
    ExtractiveQA,
    NoRAGBaseline,
    create_rag_pipeline,
    tokenize,
)


# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULL_SQUAD_PATH = DATA_DIR / "squad_dev_v2.0.json"
SUBSET_SQUAD_PATH = DATA_DIR / "squad_dev_v2.0_subset.json"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

TOKEN_PATTERN = re.compile(r"\b\w+\b")


# =============================================================================
# DATASET LOADING
# =============================================================================

def download_squad(target_path: Path) -> None:
    """Download SQuAD dataset if not present."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SQuAD dataset to {target_path}...")
    urllib.request.urlretrieve(SQUAD_URL, target_path)
    print("Download complete!")


def load_squad_dataset(
    path: Optional[Path] = None,
    max_items: int = 500,
) -> List[Dict[str, Any]]:
    """
    Load SQuAD dataset.
    
    Returns list of dicts with:
        - question: str
        - answer: str (first answer)
        - all_answers: List[str] (all valid answers)
        - context: str (paragraph context)
    """
    if path is None:
        path = FULL_SQUAD_PATH
    
    if not path.exists():
        download_squad(path)
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                
                all_answers = [a["text"] for a in answers if a.get("text")]
                if not all_answers:
                    continue
                
                items.append({
                    "question": qa["question"],
                    "answer": all_answers[0],
                    "all_answers": all_answers,
                    "context": context,
                })
                
                if len(items) >= max_items:
                    return items
    
    return items


# =============================================================================
# METRICS - FITNESS FUNCTION COMPONENTS
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    This is the PRIMARY metric (50% of fitness).
    """
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score.
    
    Returns 1.0 if normalized strings match, 0.0 otherwise.
    Small bonus (10% of fitness).
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_substring_match(prediction: str, ground_truth: str) -> float:
    """
    Compute substring match score.
    
    Returns 1.0 if one contains the other, 0.0 otherwise.
    Small bonus (10% of fitness).
    """
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not truth_norm:
        return 0.0
    
    if truth_norm in pred_norm or pred_norm in truth_norm:
        return 1.0
    return 0.0


def compute_semantic_similarity(
    prediction: str,
    ground_truth: str,
    encoder,
) -> float:
    """
    Compute semantic similarity using embeddings.
    
    Secondary metric (30% of fitness).
    """
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    
    embeddings = encoder.encode(
        [prediction, ground_truth],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    # Map from [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (similarity + 1) / 2))


def compute_fitness(
    prediction: str,
    ground_truth: str,
    all_answers: List[str],
    encoder,
) -> Dict[str, float]:
    """
    Compute complete fitness score.
    
    Fitness Function:
        fitness = 0.50 * F1_lexical
                + 0.30 * semantic_sim
                + 0.10 * exact_match
                + 0.10 * substring_match
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Primary ground truth answer
        all_answers: All valid answers for max computation
        encoder: Sentence transformer encoder for semantic similarity
    
    Returns:
        Dictionary with individual metrics and combined fitness
    """
    # Compute against best matching answer
    best_f1 = max(compute_f1(prediction, ans) for ans in all_answers)
    best_em = max(compute_exact_match(prediction, ans) for ans in all_answers)
    best_substr = max(compute_substring_match(prediction, ans) for ans in all_answers)
    semantic_sim = compute_semantic_similarity(prediction, ground_truth, encoder)
    
    # Combined fitness
    fitness = (
        0.50 * best_f1 +
        0.30 * semantic_sim +
        0.10 * best_em +
        0.10 * best_substr
    )
    
    return {
        "f1": best_f1,
        "semantic_similarity": semantic_sim,
        "exact_match": best_em,
        "substring_match": best_substr,
        "fitness": fitness,
    }


# =============================================================================
# RAG EVALUATOR CLASS
# =============================================================================

class RAGEvaluator:
    """
    Complete RAG evaluation system with caching and parallel processing.
    
    Features:
    1. LLM-based generation as primary evaluator
    2. Extractive QA baseline
    3. No-RAG LLM baseline
    4. Full config tuple caching
    5. Parallel evaluation
    """
    
    def __init__(
        self,
        settings: ExperimentSettings = DEFAULT_SETTINGS,
        use_llm: bool = True,
    ):
        self.settings = settings
        self.use_llm = use_llm
        
        # Load dataset
        print(f"Loading dataset (size={settings.dataset_size})...")
        self.dataset = load_squad_dataset(max_items=settings.dataset_size)
        print(f"Loaded {len(self.dataset)} QA pairs")
        
        # Extract documents
        self.documents = list(dict.fromkeys(
            item["context"] for item in self.dataset
        ))
        print(f"Found {len(self.documents)} unique documents")
        
        # Evaluation subset
        self.eval_sample_size = min(settings.eval_sample_size, len(self.dataset))
        self.eval_dataset = self.dataset[:self.eval_sample_size]
        print(f"Using {len(self.eval_dataset)} QA pairs per evaluation")
        
        # Pre-load LLM generator (shared across configs)
        if use_llm:
            print("Loading LLM generator (Flan-T5)...")
            self.llm_generator = LLMGenerator()
            print("LLM ready!")
        else:
            self.llm_generator = None
        
        # Load default encoder for semantic similarity
        print("Loading embedding model for evaluation...")
        from sentence_transformers import SentenceTransformer
        self.eval_encoder = SentenceTransformer(
            EMBEDDING_MODELS["minilm"]["name"]
        )
        print("Evaluator ready!")
        
        # Caching
        self._cache: Dict[Tuple, float] = {}
        self._cache_lock = Lock()
        
        # Pre-built retrievers cache (by embedding_model, chunk_size, chunk_overlap)
        self._retriever_cache: Dict[Tuple, DenseRetriever] = {}
        self._retriever_lock = Lock()
    
    def _get_retriever(
        self,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> DenseRetriever:
        """Get or create cached retriever (embeddings computed once per chunking setup)."""
        key = (embedding_model, chunk_size, chunk_overlap)
        
        with self._retriever_lock:
            if key not in self._retriever_cache:
                self._retriever_cache[key] = DenseRetriever(
                    documents=self.documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=embedding_model,
                    retrieval_metric="cosine",
                )
            return self._retriever_cache[key]
    
    def evaluate_config(
        self,
        config: RAGConfig,
        use_cache: bool = True,
    ) -> float:
        """
        Evaluate a single RAG configuration.
        
        Args:
            config: RAG configuration to evaluate
            use_cache: Whether to use cached results
        
        Returns:
            Fitness score (0.0 to 1.0)
        """
        # Include use_llm in the key so extractive baseline and generative runs cache separately
        cache_key = (self.use_llm,) + config.to_tuple()
        
        # Check cache
        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
        
        # Get or create retriever
        retriever = self._get_retriever(
            config.embedding_model,
            config.chunk_size,
            config.chunk_overlap,
        )
        
        # Create extractive QA
        extractive_qa = ExtractiveQA(encoder=retriever.encoder)
        
        # Create pipeline
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_generator=self.llm_generator if self.use_llm else None,
            extractive_qa=extractive_qa,
            retrieval_metric=config.retrieval_metric,
            max_context_chars=config.max_context_chars,
            use_llm=self.use_llm,
        )
        
        # Evaluate on sample
        scores = []
        for item in self.eval_dataset:
            prediction = pipeline.invoke(
                question=item["question"],
                top_k=config.top_k,
                similarity_threshold=config.similarity_threshold,
            )
            
            metrics = compute_fitness(
                prediction=prediction,
                ground_truth=item["answer"],
                all_answers=item.get("all_answers", [item["answer"]]),
                encoder=self.eval_encoder,
            )
            scores.append(metrics["fitness"])
        
        fitness = sum(scores) / len(scores)
        
        # Update cache
        if use_cache:
            with self._cache_lock:
                self._cache[cache_key] = fitness
        
        return fitness
    
    def evaluate_with_details(
        self,
        config: RAGConfig,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate with detailed per-question results.
        
        Returns:
            Tuple of (mean_fitness, list_of_details)
        """
        retriever = self._get_retriever(
            config.embedding_model,
            config.chunk_size,
            config.chunk_overlap,
        )
        
        extractive_qa = ExtractiveQA(encoder=retriever.encoder)
        
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_generator=self.llm_generator if self.use_llm else None,
            extractive_qa=extractive_qa,
            retrieval_metric=config.retrieval_metric,
            max_context_chars=config.max_context_chars,
            use_llm=self.use_llm,
        )
        
        details = []
        for item in self.eval_dataset:
            prediction, contexts, scores = pipeline.invoke(
                question=item["question"],
                top_k=config.top_k,
                similarity_threshold=config.similarity_threshold,
                return_details=True,
            )
            
            metrics = compute_fitness(
                prediction=prediction,
                ground_truth=item["answer"],
                all_answers=item.get("all_answers", [item["answer"]]),
                encoder=self.eval_encoder,
            )
            
            details.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                "contexts": contexts,
                "retrieval_scores": scores,
                **metrics,
            })
        
        mean_fitness = sum(d["fitness"] for d in details) / len(details)
        return mean_fitness, details
    
    def evaluate_baseline_no_rag(self) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate LLM without RAG (baseline).
        
        Shows the value added by retrieval.
        """
        if self.llm_generator is None:
            raise ValueError("LLM generator not loaded")
        
        baseline = NoRAGBaseline()
        
        details = []
        for item in self.eval_dataset:
            prediction = baseline.answer(item["question"])
            
            metrics = compute_fitness(
                prediction=prediction,
                ground_truth=item["answer"],
                all_answers=item.get("all_answers", [item["answer"]]),
                encoder=self.eval_encoder,
            )
            
            details.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                **metrics,
            })
        
        mean_fitness = sum(d["fitness"] for d in details) / len(details)
        return mean_fitness, details
    
    def clear_cache(self):
        """Clear evaluation cache."""
        with self._cache_lock:
            self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached evaluations."""
        with self._cache_lock:
            return len(self._cache)


# =============================================================================
# GLOBAL EVALUATOR (LAZY INITIALIZATION)
# =============================================================================

_GLOBAL_EVALUATOR: Optional[RAGEvaluator] = None
_INIT_LOCK = Lock()


def get_evaluator(
    settings: ExperimentSettings = DEFAULT_SETTINGS,
    use_llm: bool = True,
) -> RAGEvaluator:
    """Get or create global evaluator instance."""
    global _GLOBAL_EVALUATOR
    
    if _GLOBAL_EVALUATOR is None:
        with _INIT_LOCK:
            if _GLOBAL_EVALUATOR is None:
                _GLOBAL_EVALUATOR = RAGEvaluator(
                    settings=settings,
                    use_llm=use_llm,
                )
    
    return _GLOBAL_EVALUATOR


def evaluate_rag_pipeline(
    config_dict: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    top_k: Optional[int] = None,
    **kwargs,
) -> float:
    """
    Evaluate a RAG configuration (interface for algorithms).
    
    Args:
        config_dict: Dictionary with RAG parameters
    
    Returns:
        Fitness score
    """
    evaluator = get_evaluator()

    # Accept both modern (dict/RAGConfig) and legacy (chunk_size/top_k kwargs) usage
    if isinstance(config_dict, RAGConfig):
        config_data = config_dict.to_dict()
    elif isinstance(config_dict, dict) and config_dict is not None:
        config_data = dict(config_dict)
    else:
        config_data = {}

    if chunk_size is not None:
        config_data["chunk_size"] = chunk_size
    if top_k is not None:
        config_data["top_k"] = top_k

    if kwargs:
        config_data.update(kwargs)

    config = RAGConfig.from_dict(config_data)
    return evaluator.evaluate_config(config)


# =============================================================================
# LEGACY INTERFACE (for backward compatibility)
# =============================================================================

def evaluate_rag_pipeline_legacy(
    chunk_size: int,
    top_k: int,
    **kwargs,
) -> float:
    """Legacy interface for old algorithm code."""
    config = RAGConfig(
        chunk_size=chunk_size,
        top_k=top_k,
        chunk_overlap=kwargs.get("chunk_overlap", 64),
        similarity_threshold=kwargs.get("similarity_threshold", 0.2),
        retrieval_metric=kwargs.get("retrieval_metric", "cosine"),
        embedding_model=kwargs.get("embedding_model", "minilm"),
        max_context_chars=kwargs.get("max_context_chars", 2048),
    )
    evaluator = get_evaluator()
    return evaluator.evaluate_config(config)
