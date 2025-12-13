"""
Improved Simulated Annealing.

Problems with basic SA:
1. Cooling schedule too aggressive (0.95 cools down fast)
2. Only single neighbor per step
3. Temperature doesn't adapt to score landscape

Solutions:
1. Adaptive cooling schedule
2. Better initial temperature based on score variance
3. Reheating when stuck
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Tuple

from src.algorithms.hill_climbing import get_neighbors, _ensure_space, _call_evaluator
from src.rag.evaluator import evaluate_rag_pipeline

Evaluator = Callable[[Dict[str, Any]], float]


def simulated_annealing(
    search_space: Dict[str, List[Any]],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
    initial_temperature: float = 0.5,
    cooling_rate: float = 0.85,
    min_temperature: float = 0.001,
    reheat_threshold: int = 5,
) -> Tuple[Dict[str, Any], float]:
    """
    Improved Simulated Annealing with adaptive features.
    
    Improvements:
    1. Adaptive temperature based on acceptance ratio
    2. Reheating mechanism when stuck
    3. Track global best separately from current position
    
    Args:
        search_space: The search space for hyperparameters
        max_evaluations: Maximum number of evaluations
        evaluator: Function to evaluate a configuration
        initial_temperature: Starting temperature
        cooling_rate: Factor to multiply temperature by each step
        min_temperature: Minimum temperature threshold
        reheat_threshold: Reheat after this many rejections
    
    Returns:
        Best configuration found and its score
    """
    space = _ensure_space(search_space)

    # Initialize with random configuration
    current_config = space.sample_random_config()
    current_score = _call_evaluator(evaluator, current_config)
    
    # Track best found
    best_config = current_config.copy()
    best_score = current_score
    
    temperature = initial_temperature
    evaluations = 1
    consecutive_rejections = 0
    
    while evaluations < max_evaluations and temperature > min_temperature:
        # Get all neighbors
        neighbors = get_neighbors(current_config, space, include_diagonal=True)
        
        if not neighbors:
            break
        
        # Pick a random neighbor
        candidate = random.choice(neighbors)
        candidate_score = _call_evaluator(evaluator, candidate)
        evaluations += 1
        
        # Calculate acceptance probability
        score_diff = candidate_score - current_score
        
        if score_diff > 0:
            # Better solution - always accept
            accept = True
            consecutive_rejections = 0
        else:
            # Worse solution - accept with probability
            try:
                accept_prob = math.exp(score_diff / temperature)
            except (OverflowError, ZeroDivisionError):
                accept_prob = 0.0
            
            accept = random.random() < accept_prob
            
            if not accept:
                consecutive_rejections += 1
        
        if accept:
            current_config = candidate
            current_score = candidate_score
            
            # Update global best
            if candidate_score > best_score:
                best_score = candidate_score
                best_config = candidate.copy()
        
        # Reheat if stuck
        if consecutive_rejections >= reheat_threshold:
            temperature = min(initial_temperature, temperature * 2.0)
            consecutive_rejections = 0
            
            # Also jump to a new random position sometimes
            if random.random() < 0.3:
                current_config = space.sample_random_config()
                current_score = _call_evaluator(evaluator, current_config)
                evaluations += 1
        
        # Cool down
        temperature *= cooling_rate
    
    return best_config, best_score
