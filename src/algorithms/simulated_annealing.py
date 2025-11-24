import math
import random
from typing import Callable, Dict, List, Tuple

from src.algorithms.hill_climbing import get_neighbors
from src.rag.evaluator import evaluate_rag_pipeline

Evaluator = Callable[[int, int], float]


def simulated_annealing(
    search_space: Dict[str, List[int]],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 1e-3,
) -> Tuple[Dict[str, int], float]:
    """
    Performs simulated annealing to find the best RAG configuration.

    Args:
        search_space (dict): The search space for the hyperparameters.
        max_evaluations (int): The maximum number of evaluations to perform.
        evaluator (callable): Function to evaluate a configuration and return a score.
        initial_temperature (float): Starting temperature that controls exploration.
        cooling_rate (float): Multiplicative factor to cool the temperature each step.
        min_temperature (float): Temperature threshold to stop early.

    Returns:
        A tuple containing the best configuration found and its score.
    """
    current_config = {
        "chunk_size": random.choice(search_space["chunk_size"]),
        "top_k": random.choice(search_space["top_k"]),
    }
    current_score = evaluator(current_config["chunk_size"], current_config["top_k"])
    best_config = current_config
    best_score = current_score

    temperature = max(initial_temperature, min_temperature)
    evaluations = 1

    while evaluations < max_evaluations and temperature > min_temperature:
        neighbors = get_neighbors(current_config, search_space)
        if not neighbors:
            break

        candidate = random.choice(neighbors)
        candidate_score = evaluator(candidate["chunk_size"], candidate["top_k"])
        evaluations += 1

        score_diff = candidate_score - current_score
        accept_probability = 1.0
        if score_diff <= 0:
            # Allow occasional downhill moves to escape local optima.
            accept_probability = math.exp(score_diff / max(temperature, 1e-8))

        if score_diff > 0 or random.random() < accept_probability:
            current_config = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best_config = candidate
                best_score = candidate_score

        temperature *= cooling_rate

    return best_config, best_score
