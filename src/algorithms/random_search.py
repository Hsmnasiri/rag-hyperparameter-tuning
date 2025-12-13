import random
from typing import Any, Callable, Dict, Tuple

from src.rag.evaluator import evaluate_rag_pipeline
from src.rag.search_space import SearchSpace

Evaluator = Callable[[Dict[str, Any]], float]


def _ensure_space(search_space: Dict[str, list]) -> SearchSpace:
    """Normalize plain dict space into a SearchSpace object."""
    if isinstance(search_space, SearchSpace):
        return search_space
    return SearchSpace.from_dict(search_space)


def _call_evaluator(evaluator: Evaluator, config: Dict[str, Any]) -> float:
    """Call evaluator with backward-compatible fallbacks."""
    try:
        return evaluator(config)  # type: ignore[arg-type]
    except TypeError:
        return evaluator(config.get("chunk_size"), config.get("top_k"))  # type: ignore[misc]


def random_search(
    search_space: Dict[str, list],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
) -> Tuple[Dict[str, Any], float]:
    """
    Performs random search to find the best RAG configuration.

    Args:
        search_space (dict): The search space for the hyperparameters.
        max_evaluations (int): The maximum number of evaluations to perform.

    Returns:
        A tuple containing the best configuration found and its score.
    """
    space = _ensure_space(search_space)

    # Default budget aligns with requested 20-30 evaluations when not specified
    if max_evaluations is None or max_evaluations <= 0:
        max_evaluations = min(30, max(20, space.get_total_configurations() // 10))

    best_config = None
    best_score = -1

    for _ in range(max_evaluations):
        # Sample a random configuration and ensure it is valid
        config = space.sample_random_config()
        is_valid, _ = space.validate_config(config)
        if not is_valid:
            continue

        # Evaluate the configuration
        score = _call_evaluator(evaluator, config)

        # Update the best configuration if the current one is better
        if score > best_score:
            best_score = score
            best_config = config

    if best_config is None:
        best_config = space.sample_random_config()
        best_score = _call_evaluator(evaluator, best_config)

    return best_config, best_score
