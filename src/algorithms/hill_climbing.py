"""
Improved Hill Climbing with Random Restarts.

Problems with basic Hill Climbing:
1. Gets stuck in local optima
2. Only explores neighbors, misses good regions

Solutions:
1. Random restarts when stuck
2. Steepest ascent (evaluate all neighbors)
3. Better neighbor generation (diagonal moves)
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

from src.rag.evaluator import evaluate_rag_pipeline
from src.rag.search_space import SearchSpace

Evaluator = Callable[[Dict[str, Any]], float]


def _ensure_space(search_space: Dict[str, List[Any]]) -> SearchSpace:
    """Normalize plain dict space into a SearchSpace object."""
    if isinstance(search_space, SearchSpace):
        return search_space
    return SearchSpace.from_dict(search_space)


def _call_evaluator(evaluator: Evaluator, config: Dict[str, Any]) -> float:
    """
    Call evaluator with a config dict while preserving legacy (chunk_size, top_k) evaluators.
    """
    try:
        return evaluator(config)  # type: ignore[arg-type]
    except TypeError:
        return evaluator(config.get("chunk_size"), config.get("top_k"))  # type: ignore[misc]


def get_neighbors(
    config: Dict[str, Any], 
    search_space: Dict[str, List[Any]],
    include_diagonal: bool = True,
) -> List[Dict[str, int]]:
    """
    Generate neighbors of a configuration.
    
    Args:
        config: Current configuration
        search_space: The search space
        include_diagonal: If True, include diagonal moves (both params change)
    
    Returns:
        List of neighboring configurations
    """
    space = _ensure_space(search_space)
    neighbors = space.get_neighbors(config, include_diagonal=include_diagonal)

    # Filter invalid combos (e.g., overlap >= chunk_size)
    valid_neighbors = []
    for neighbor in neighbors:
        is_valid, _ = space.validate_config(neighbor)
        if is_valid and neighbor not in valid_neighbors:
            valid_neighbors.append(neighbor)
    return valid_neighbors


def hill_climbing(
    search_space: Dict[str, List[Any]],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
    num_restarts: int = 3,
    include_diagonal: bool = True,
) -> Tuple[Dict[str, int], float]:
    """
    Hill Climbing with Random Restarts.
    
    This version includes:
    1. Multiple random restarts to escape local optima
    2. Steepest ascent - evaluates all neighbors and picks the best
    3. Diagonal moves for faster exploration
    
    Args:
        search_space: The search space for hyperparameters
        max_evaluations: Maximum number of evaluations
        evaluator: Function to evaluate a configuration
        num_restarts: Number of random restarts to perform
        include_diagonal: Whether to include diagonal neighbor moves
    
    Returns:
        Best configuration found and its score
    """
    space = _ensure_space(search_space)
    best_overall_config: Dict[str, Any] | None = None
    best_overall_score = -1.0
    evaluations = 0

    evals_per_restart = max(1, max_evaluations // max(1, num_restarts))

    for _ in range(num_restarts):
        if evaluations >= max_evaluations:
            break

        # Sample a valid starting point
        attempts = 0
        current_config = space.sample_random_config()
        is_valid, _ = space.validate_config(current_config)
        while not is_valid and attempts < 5:
            current_config = space.sample_random_config()
            is_valid, _ = space.validate_config(current_config)
            attempts += 1

        current_score = _call_evaluator(evaluator, current_config)
        evaluations += 1

        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_config = current_config.copy()

        restart_evals = 1
        improved = True

        while improved and evaluations < max_evaluations and restart_evals < evals_per_restart:
            improved = False
            neighbors = get_neighbors(current_config, space, include_diagonal)

            best_neighbor = None
            best_neighbor_score = current_score

            # Steepest ascent: evaluate every neighbor and move to the best
            for neighbor in neighbors:
                if evaluations >= max_evaluations or restart_evals >= evals_per_restart:
                    break

                score = _call_evaluator(evaluator, neighbor)
                evaluations += 1
                restart_evals += 1

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor

                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_config = neighbor.copy()

            if best_neighbor is not None and best_neighbor_score > current_score:
                current_config = best_neighbor
                current_score = best_neighbor_score
                improved = True

    # Fallback to a sampled config if nothing evaluated
    if best_overall_config is None:
        best_overall_config = space.sample_random_config()
        best_overall_score = _call_evaluator(evaluator, best_overall_config)

    return best_overall_config, best_overall_score
