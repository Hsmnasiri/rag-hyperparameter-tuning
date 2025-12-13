"""
RAG Fitness Function and Evaluation Metrics.

Based on academic standards:
- F1 Score (primary signal) - 50%
- Semantic Similarity - 30%
- Exact Match bonus - 10%
- Substring Match bonus - 10%

References:
- RAGAS (Es et al., 2023)
- RAGBench (Belyi et al., 2024)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return TOKEN_PATTERN.findall(text.lower())


def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison.
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Normalize whitespace
    """
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    This is the PRIMARY evaluation metric as per academic standards.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Where:
    - precision = |pred ∩ truth| / |pred|
    - recall = |pred ∩ truth| / |truth|
    """
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    pred_set = set(pred_tokens)
    truth_set = set(truth_tokens)
    common = pred_set & truth_set
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score (0 or 1).
    
    Returns 1.0 if normalized prediction equals normalized ground truth.
    """
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == truth_norm else 0.0


def compute_substring_match(prediction: str, ground_truth: str) -> float:
    """
    Compute substring match score (0 or 1).
    
    Returns 1.0 if:
    - Ground truth is contained in prediction, OR
    - Prediction is contained in ground truth
    """
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not truth_norm:
        return 0.0
    
    if truth_norm in pred_norm or pred_norm in truth_norm:
        return 1.0
    return 0.0


def compute_semantic_similarity(
    pred_embedding: np.ndarray,
    truth_embedding: np.ndarray,
) -> float:
    """
    Compute semantic similarity using cosine similarity.
    
    Maps from [-1, 1] to [0, 1] for consistency.
    """
    if pred_embedding is None or truth_embedding is None:
        return 0.0
    
    # Ensure normalized
    pred_norm = pred_embedding / (np.linalg.norm(pred_embedding) + 1e-8)
    truth_norm = truth_embedding / (np.linalg.norm(truth_embedding) + 1e-8)
    
    similarity = float(np.dot(pred_norm, truth_norm))
    
    # Map [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (similarity + 1) / 2))


@dataclass
class EvaluationResult:
    """
    Detailed evaluation result for a single QA pair.
    """
    question: str
    ground_truth: str
    prediction: str
    
    # Individual metrics
    f1_score: float
    exact_match: float
    substring_match: float
    semantic_similarity: float
    
    # Combined fitness
    fitness: float
    
    # Optional: retrieved contexts
    contexts: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "substring_match": self.substring_match,
            "semantic_similarity": self.semantic_similarity,
            "fitness": self.fitness,
        }


def compute_fitness(
    prediction: str,
    ground_truth: str,
    pred_embedding: Optional[np.ndarray] = None,
    truth_embedding: Optional[np.ndarray] = None,
    all_ground_truths: Optional[List[str]] = None,
) -> tuple:
    """
    Compute the combined fitness score.
    
    Fitness Function (as specified):
        fitness = 0.5 * F1_lexical
                + 0.3 * semantic_similarity
                + 0.1 * exact_match_bonus
                + 0.1 * substring_bonus
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Primary ground truth answer
        pred_embedding: Embedding of prediction (optional)
        truth_embedding: Embedding of ground truth (optional)
        all_ground_truths: List of all valid ground truths (for multi-answer QA)
    
    Returns:
        Tuple of (fitness, f1, semantic_sim, exact_match, substring_match)
    """
    # If multiple ground truths, compute against best match
    if all_ground_truths and len(all_ground_truths) > 1:
        best_f1 = max(compute_f1(prediction, gt) for gt in all_ground_truths)
        best_exact = max(compute_exact_match(prediction, gt) for gt in all_ground_truths)
        best_substring = max(compute_substring_match(prediction, gt) for gt in all_ground_truths)
    else:
        best_f1 = compute_f1(prediction, ground_truth)
        best_exact = compute_exact_match(prediction, ground_truth)
        best_substring = compute_substring_match(prediction, ground_truth)
    
    # Semantic similarity
    if pred_embedding is not None and truth_embedding is not None:
        semantic_sim = compute_semantic_similarity(pred_embedding, truth_embedding)
    else:
        # Fallback: use F1 as proxy if no embeddings
        semantic_sim = best_f1
    
    # Combined fitness with specified weights
    fitness = (
        0.5 * best_f1 +
        0.3 * semantic_sim +
        0.1 * best_exact +
        0.1 * best_substring
    )
    
    return fitness, best_f1, semantic_sim, best_exact, best_substring


class FitnessCalculator:
    """
    Calculator for fitness scores with embedding support.
    
    This class handles:
    - Embedding computation for semantic similarity
    - Batch processing for efficiency
    - Consistent scoring across all algorithms
    """
    
    def __init__(self, encoder=None):
        """
        Args:
            encoder: SentenceTransformer encoder for semantic similarity.
                    If None, semantic similarity will fallback to F1.
        """
        self.encoder = encoder
    
    def compute_single(
        self,
        prediction: str,
        ground_truth: str,
        all_ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Compute fitness for a single QA pair."""
        # Compute embeddings if encoder available
        pred_emb = None
        truth_emb = None
        
        if self.encoder is not None:
            try:
                embeddings = self.encoder.encode(
                    [prediction, ground_truth],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                pred_emb = embeddings[0]
                truth_emb = embeddings[1]
            except Exception:
                pass
        
        fitness, f1, semantic, exact, substring = compute_fitness(
            prediction=prediction,
            ground_truth=ground_truth,
            pred_embedding=pred_emb,
            truth_embedding=truth_emb,
            all_ground_truths=all_ground_truths,
        )
        
        return EvaluationResult(
            question="",  # To be filled by caller
            ground_truth=ground_truth,
            prediction=prediction,
            f1_score=f1,
            exact_match=exact,
            substring_match=substring,
            semantic_similarity=semantic,
            fitness=fitness,
        )
    
    def compute_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        all_ground_truths_list: Optional[List[List[str]]] = None,
    ) -> List[EvaluationResult]:
        """Compute fitness for a batch of QA pairs (more efficient)."""
        if not predictions:
            return []
        
        # Batch encode if encoder available
        pred_embs = None
        truth_embs = None
        
        if self.encoder is not None:
            try:
                all_texts = predictions + ground_truths
                embeddings = self.encoder.encode(
                    all_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64,
                )
                n = len(predictions)
                pred_embs = embeddings[:n]
                truth_embs = embeddings[n:]
            except Exception:
                pass
        
        results = []
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            pred_emb = pred_embs[i] if pred_embs is not None else None
            truth_emb = truth_embs[i] if truth_embs is not None else None
            all_gts = all_ground_truths_list[i] if all_ground_truths_list else None
            
            fitness, f1, semantic, exact, substring = compute_fitness(
                prediction=pred,
                ground_truth=truth,
                pred_embedding=pred_emb,
                truth_embedding=truth_emb,
                all_ground_truths=all_gts,
            )
            
            results.append(EvaluationResult(
                question="",
                ground_truth=truth,
                prediction=pred,
                f1_score=f1,
                exact_match=exact,
                substring_match=substring,
                semantic_similarity=semantic,
                fitness=fitness,
            ))
        
        return results
