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
import random
import re
import urllib.request
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    RAGConfig,
    ExperimentSettings,
    DEFAULT_SETTINGS,
    EMBEDDING_MODELS,
    GENERATOR_MODELS,
)
from .pipeline import (
    RAGPipeline,
    DenseRetriever,
    TfidfRetriever,
    LLMGenerator,
    ExtractiveQA,
    NoRAGBaseline,
    resolve_device,
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

_ALLOWED_RETRIEVERS = {"dense", "tfidf"}


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def resolve_dataset_path() -> Path:
    """
    Resolve dataset path with a sensible fallback chain.

    Priority:
    1) RAG_DATASET_PATH (if set)
    2) data/squad_dev_v2.0.json (download if missing)
    3) data/squad_dev_v2.0_subset.json (if present)
    """
    raw = os.environ.get("RAG_DATASET_PATH")
    target = Path(raw).expanduser() if raw else FULL_SQUAD_PATH

    if target.exists():
        return target

    try:
        download_squad(target)
        return target
    except Exception:
        if SUBSET_SQUAD_PATH.exists():
            return SUBSET_SQUAD_PATH
        raise


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
    max_items: Optional[int] = 500,
    seed: Optional[int] = None,
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
        path = resolve_dataset_path()
    
    if not path.exists():
        download_squad(path)
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    items: List[Dict[str, Any]] = []
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

                # Fast-path: preserve old behavior when no seed is requested
                if seed is None and max_items is not None and len(items) >= max_items:
                    return items

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)

    if max_items is None:
        return items
    return items[:max_items]


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
    encoder: Optional[Any],
) -> float:
    """
    Compute semantic similarity using embeddings.
    
    Secondary metric (30% of fitness).
    """
    if encoder is None:
        return 0.0
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
    encoder: Optional[Any],
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
    semantic_sim = (
        compute_semantic_similarity(prediction, ground_truth, encoder)
        if encoder is not None
        else best_f1
    )
    
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
        retriever_type: str = "dense",
        enable_semantic_similarity: bool = True,
        generator_model: str = "google/flan-t5-small",
        generator_max_new_tokens: int = 64,
        generator_temperature: float = 0.0,
        device: Optional[str] = None,
    ):
        self.settings = settings
        self.use_llm = use_llm
        self.retriever_type = retriever_type if retriever_type in _ALLOWED_RETRIEVERS else "dense"
        self.enable_semantic_similarity = enable_semantic_similarity
        self.num_workers = settings.num_workers
        self.device = resolve_device(device)
        try:
            import torch

            if self.device == "cpu" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                print("Note: MPS is available but CPU is selected. Set RAG_DEVICE=mps to use Apple Silicon acceleration.")
        except Exception:
            pass
        self.generator_model = generator_model
        self.generator_max_new_tokens = generator_max_new_tokens
        self.generator_temperature = generator_temperature

        # Trace / per-question logging (optional; can be large)
        self.trace_qa = _env_bool("RAG_TRACE_QA", False)
        self.trace_baseline = _env_bool("RAG_TRACE_BASELINE", False)
        self.trace_max_contexts = max(1, _env_int("RAG_TRACE_MAX_CONTEXTS", 5))
        self.trace_max_context_chars = max(0, _env_int("RAG_TRACE_MAX_CONTEXT_CHARS", 300))
        self.trace_max_prompt_chars = max(0, _env_int("RAG_TRACE_MAX_PROMPT_CHARS", 2000))
        self.trace_max_prediction_chars = max(0, _env_int("RAG_TRACE_MAX_PREDICTION_CHARS", 500))
        self.trace_sample_size = max(0, _env_int("RAG_TRACE_SAMPLE_SIZE", 0))
        results_dir = Path(os.environ.get("RAG_RESULTS_DIR", "results")).expanduser()
        self.trace_path = results_dir / "live" / "qa_traces.jsonl"
        self._trace_lock = Lock()
        if self.trace_qa or self.trace_baseline:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Load dataset
        dataset_path = resolve_dataset_path()
        print(f"Loading dataset from {dataset_path} (size={settings.dataset_size})...")
        self.dataset = load_squad_dataset(
            path=dataset_path,
            max_items=settings.dataset_size,
            seed=settings.dataset_seed,
        )
        print(f"Loaded {len(self.dataset)} QA pairs")
        
        # Extract documents
        self.documents = list(dict.fromkeys(
            item["context"] for item in self.dataset
        ))
        print(f"Found {len(self.documents)} unique documents")
        
        # Evaluation subset
        self.eval_sample_size = min(settings.eval_sample_size, len(self.dataset))
        if self.eval_sample_size >= len(self.dataset):
            self.eval_dataset = list(self.dataset)
        else:
            rng = random.Random(settings.dataset_seed + 1)
            self.eval_dataset = rng.sample(self.dataset, k=self.eval_sample_size)
        print(f"Using {len(self.eval_dataset)} QA pairs per evaluation")
        
        # Pre-load LLM generator (shared across configs)
        if use_llm:
            print(f"Loading LLM generator ({generator_model})...")
            self.llm_generator = LLMGenerator(
                model_name=generator_model,
                max_new_tokens=generator_max_new_tokens,
                temperature=generator_temperature,
                device=self.device,
            )
            print("LLM ready!")
        else:
            self.llm_generator = None
        
        # Load default encoder for semantic similarity (optional)
        if self.enable_semantic_similarity:
            print("Loading embedding model for evaluation...")
            from sentence_transformers import SentenceTransformer
            self.eval_encoder = SentenceTransformer(
                EMBEDDING_MODELS["minilm"]["name"],
                device=self.device,
            )
        else:
            self.eval_encoder = None
        print("Evaluator ready!")
        
        # Caching
        self._cache: Dict[Tuple, float] = {}
        self._cache_lock = Lock()
        
        # Pre-built retrievers cache (by embedding_model, chunk_size, chunk_overlap)
        self._retriever_cache: Dict[Tuple, Any] = {}
        self._retriever_lock = Lock()

    def _log_trace(self, record: Dict[str, Any]) -> None:
        if not (self.trace_qa or self.trace_baseline):
            return
        path_override = record.pop("_trace_file", None)
        target_path = Path(path_override).expanduser() if path_override else self.trace_path
        try:
            line = json.dumps(record, ensure_ascii=False)
        except TypeError:
            # Best-effort fallback for non-serializable objects
            safe = {k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v) for k, v in record.items()}
            line = json.dumps(safe, ensure_ascii=False)
        with self._trace_lock:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    
    def _get_retriever(
        self,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Any:
        """Get or create cached retriever (computed once per chunking setup)."""
        if self.retriever_type == "tfidf":
            key = (self.retriever_type, chunk_size, chunk_overlap)
        else:
            key = (self.retriever_type, embedding_model, chunk_size, chunk_overlap)
        
        with self._retriever_lock:
            if key not in self._retriever_cache:
                if self.retriever_type == "tfidf":
                    self._retriever_cache[key] = TfidfRetriever(
                        documents=self.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                else:
                    self._retriever_cache[key] = DenseRetriever(
                        documents=self.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embedding_model=embedding_model,
                        retrieval_metric="cosine",
                        device=self.device,
                    )
            return self._retriever_cache[key]
    
    def evaluate_config(
        self,
        config: RAGConfig,
        use_cache: bool = True,
        progress: Optional[Callable[[int, int], None]] = None,
        trace: Optional[Dict[str, Any]] = None,
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
        cache_key = (
            self.retriever_type,
            self.enable_semantic_similarity,
            self.use_llm,
        ) + config.to_tuple()
        
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
        total_items = len(self.eval_dataset)
        sample_indices = None
        if self.trace_qa and trace is not None and self.trace_sample_size > 0 and total_items > 0:
            k = min(self.trace_sample_size, total_items)
            sample_indices = set(random.sample(range(1, total_items + 1), k=k))
        for idx, item in enumerate(self.eval_dataset, start=1):
            contexts: List[str] = []
            retrieval_scores: List[float] = []
            prompt_text: Optional[str] = None

            if self.trace_qa and trace is not None:
                prediction, contexts, retrieval_scores, prompt_text = pipeline.invoke(
                    question=item["question"],
                    top_k=config.top_k,
                    similarity_threshold=config.similarity_threshold,
                    return_details=True,
                    return_prompt=True,
                )
            else:
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

            if self.trace_qa and trace is not None:
                if sample_indices is not None and idx not in sample_indices:
                    if progress is not None and (idx == 1 or idx % 10 == 0 or idx == total_items):
                        progress(idx, total_items)
                    continue
                trimmed_contexts = [
                    _truncate_text(c, self.trace_max_context_chars)
                    for c in contexts[: self.trace_max_contexts]
                ]
                trimmed_prompt = (
                    _truncate_text(prompt_text, self.trace_max_prompt_chars)
                    if prompt_text is not None
                    else None
                )
                trimmed_pred = _truncate_text(prediction, self.trace_max_prediction_chars)
                self._log_trace(
                    {
                        "kind": "rag_item",
                        "trace": trace,
                        "item": idx,
                        "total_items": total_items,
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "all_answers": item.get("all_answers", [item["answer"]]),
                        "config": config.to_dict(),
                        "retriever_type": self.retriever_type,
                        "retrieval_metric": config.retrieval_metric,
                        "top_k": int(config.top_k),
                        "similarity_threshold": float(config.similarity_threshold),
                        "retrieved_contexts": trimmed_contexts,
                        "retrieval_scores": [float(s) for s in retrieval_scores[: len(trimmed_contexts)]],
                        "prompt": trimmed_prompt,
                        "prediction": trimmed_pred,
                        **metrics,
                        "_trace_file": trace.get("trace_file") if isinstance(trace, dict) else None,
                    }
                )
            if progress is not None and (idx == 1 or idx % 10 == 0 or idx == total_items):
                progress(idx, total_items)
        
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
    
    def evaluate_baseline_no_rag(
        self,
        return_details: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate LLM without RAG (baseline).
        
        Shows the value added by retrieval.
        """
        if self.llm_generator is None:
            raise ValueError("LLM generator not loaded")
        
        details: List[Dict[str, Any]] = []
        scores: List[float] = []
        total_items = len(self.eval_dataset)
        for idx, item in enumerate(self.eval_dataset, start=1):
            prompt = f"Answer the following question:\n\nQuestion: {item['question']}\n\nAnswer:"
            prediction = self.llm_generator.generate_prompt(prompt)
            
            metrics = compute_fitness(
                prediction=prediction,
                ground_truth=item["answer"],
                all_answers=item.get("all_answers", [item["answer"]]),
                encoder=self.eval_encoder,
            )
            scores.append(metrics["fitness"])

            if return_details:
                details.append({
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "prediction": prediction,
                    **metrics,
                })

            if self.trace_baseline and trace is not None:
                self._log_trace(
                    {
                        "kind": "baseline_no_rag_item",
                        "trace": trace,
                        "item": idx,
                        "total_items": total_items,
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "all_answers": item.get("all_answers", [item["answer"]]),
                        "prompt": _truncate_text(prompt, self.trace_max_prompt_chars),
                        "prediction": _truncate_text(prediction, self.trace_max_prediction_chars),
                        **metrics,
                    }
                )

            if progress is not None and (idx == 1 or idx % 10 == 0 or idx == total_items):
                progress(idx, total_items)

        mean_fitness = sum(scores) / len(scores)
        return mean_fitness, details
    
    def clear_cache(self):
        """Clear evaluation cache."""
        with self._cache_lock:
            self._cache.clear()

    def clear_all_caches(self):
        """Clear evaluation + retriever caches."""
        with self._cache_lock:
            self._cache.clear()
        with self._retriever_lock:
            self._retriever_cache.clear()
    
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
    use_llm: Optional[bool] = None,
    retriever_type: Optional[str] = None,
    enable_semantic_similarity: Optional[bool] = None,
) -> RAGEvaluator:
    """Get or create global evaluator instance."""
    global _GLOBAL_EVALUATOR
    
    if _GLOBAL_EVALUATOR is None:
        with _INIT_LOCK:
            if _GLOBAL_EVALUATOR is None:
                # Allow environment variables to override defaults.
                if settings is DEFAULT_SETTINGS:
                    settings = replace(
                        DEFAULT_SETTINGS,
                        dataset_size=_env_int("RAG_DATASET_SIZE", DEFAULT_SETTINGS.dataset_size),
                        eval_sample_size=_env_int("RAG_EVAL_SAMPLE_SIZE", DEFAULT_SETTINGS.eval_sample_size),
                        dataset_seed=_env_int("RAG_DATASET_SEED", DEFAULT_SETTINGS.dataset_seed),
                        num_workers=_env_int("RAG_NUM_WORKERS", DEFAULT_SETTINGS.num_workers),
                    )

                if retriever_type is None:
                    retriever_type = os.environ.get("RAG_RETRIEVER", "dense").strip().lower()
                if use_llm is None:
                    generator_mode = os.environ.get("RAG_GENERATOR", "seq2seq").strip().lower()
                    use_llm = generator_mode != "extractive"

                if enable_semantic_similarity is None:
                    default_sem = (retriever_type != "tfidf")
                    enable_semantic_similarity = _env_bool("RAG_ENABLE_SEMANTIC_SIM", default_sem)

                generator_model_env = os.environ.get("RAG_GENERATOR_MODEL", "flan-t5-small").strip()
                generator_model = GENERATOR_MODELS.get(generator_model_env, {}).get(
                    "name", generator_model_env
                )

                generator_max_new_tokens = _env_int("RAG_GENERATOR_MAX_NEW_TOKENS", 64)
                generator_temperature = _env_float("RAG_GENERATOR_TEMPERATURE", 0.0)
                device = os.environ.get("RAG_DEVICE")

                _GLOBAL_EVALUATOR = RAGEvaluator(
                    settings=settings,
                    use_llm=bool(use_llm),
                    retriever_type=retriever_type,
                    enable_semantic_similarity=bool(enable_semantic_similarity),
                    generator_model=generator_model,
                    generator_max_new_tokens=generator_max_new_tokens,
                    generator_temperature=generator_temperature,
                    device=device,
                )
    
    return _GLOBAL_EVALUATOR


def evaluate_rag_pipeline(
    config_dict: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    top_k: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    trace: Optional[Dict[str, Any]] = None,
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
    return evaluator.evaluate_config(config, progress=progress, trace=trace)


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
