import random
from typing import Callable, Dict, Tuple

from src.rag.evaluator import evaluate_rag_pipeline

Evaluator = Callable[[int, int], float]


def random_search(
    search_space: Dict[str, list],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
) -> Tuple[Dict[str, int], float]:
    """
    Performs random search to find the best RAG configuration.

    Args:
        search_space (dict): The search space for the hyperparameters.
        max_evaluations (int): The maximum number of evaluations to perform.

    Returns:
        A tuple containing the best configuration found and its score.
    """
    best_config = None
    best_score = -1

    for _ in range(max_evaluations):
        # Sample a random configuration
        chunk_size = random.choice(search_space["chunk_size"])
        top_k = random.choice(search_space["top_k"])
        config = {"chunk_size": chunk_size, "top_k": top_k}

        # Evaluate the configuration
        score = evaluator(chunk_size, top_k)

        # Update the best configuration if the current one is better
        if score > best_score:
            best_score = score
            best_config = config

    return best_config, best_score
