import random
from typing import Callable, Dict, List, Tuple

from src.rag.evaluator import evaluate_rag_pipeline

Evaluator = Callable[[int, int], float]


def get_neighbors(config: Dict[str, int], search_space: Dict[str, List[int]]) -> List[Dict[str, int]]:
    """
    Generates neighbors of a given configuration.

    Args:
        config (dict): The configuration to generate neighbors for.
        search_space (dict): The search space for the hyperparameters.

    Returns:
        A list of neighboring configurations.
    """
    neighbors = []
    
    # Add neighbors for chunk_size
    current_chunk_size_index = search_space["chunk_size"].index(config["chunk_size"])
    if current_chunk_size_index > 0:
        new_chunk_size = search_space["chunk_size"][current_chunk_size_index - 1]
        neighbors.append({"chunk_size": new_chunk_size, "top_k": config["top_k"]})
    if current_chunk_size_index < len(search_space["chunk_size"]) - 1:
        new_chunk_size = search_space["chunk_size"][current_chunk_size_index + 1]
        neighbors.append({"chunk_size": new_chunk_size, "top_k": config["top_k"]})

    # Add neighbors for top_k
    current_top_k_index = search_space["top_k"].index(config["top_k"])
    if current_top_k_index > 0:
        new_top_k = search_space["top_k"][current_top_k_index - 1]
        neighbors.append({"chunk_size": config["chunk_size"], "top_k": new_top_k})
    if current_top_k_index < len(search_space["top_k"]) - 1:
        new_top_k = search_space["top_k"][current_top_k_index + 1]
        neighbors.append({"chunk_size": config["chunk_size"], "top_k": new_top_k})
        
    return neighbors


def hill_climbing(
    search_space: Dict[str, List[int]],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
) -> Tuple[Dict[str, int], float]:
    """
    Performs hill climbing to find the best RAG configuration.

    Args:
        search_space (dict): The search space for the hyperparameters.
        max_evaluations (int): The maximum number of evaluations to perform.

    Returns:
        A tuple containing the best configuration found and its score.
    """
    # Start with a random configuration
    current_chunk_size = random.choice(search_space["chunk_size"])
    current_top_k = random.choice(search_space["top_k"])
    current_config = {"chunk_size": current_chunk_size, "top_k": current_top_k}
    current_score = evaluator(current_chunk_size, current_top_k)
    
    evaluations = 1
    while evaluations < max_evaluations:
        # Generate neighbors
        neighbors = get_neighbors(current_config, search_space)
        
        # Find the best neighbor
        best_neighbor = None
        best_neighbor_score = -1
        for neighbor in neighbors:
            if evaluations >= max_evaluations:
                break
            
            score = evaluator(neighbor["chunk_size"], neighbor["top_k"])
            evaluations += 1
            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor
        
        # If the best neighbor is better than the current configuration, move to it
        if best_neighbor_score > current_score:
            current_config = best_neighbor
            current_score = best_neighbor_score
        else:
            # If no neighbor is better, we have reached a local maximum
            break
            
    return current_config, current_score
