"""
Genetic Algorithm for RAG Hyperparameter Optimization.

This implements a simple GA as mentioned in the proposal's reference to
Bulhakov et al. (2025) and NSGA-II for LLM tuning problems.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.rag.evaluator import evaluate_rag_pipeline

Evaluator = Callable[[int, int], float]


@dataclass
class Individual:
    """Represents a single configuration in the population."""
    chunk_size: int
    top_k: int
    fitness: float = 0.0

    def to_dict(self) -> Dict[str, int]:
        return {"chunk_size": self.chunk_size, "top_k": self.top_k}


def create_individual(search_space: Dict[str, List[int]]) -> Individual:
    """Create a random individual from the search space."""
    return Individual(
        chunk_size=random.choice(search_space["chunk_size"]),
        top_k=random.choice(search_space["top_k"]),
    )


def evaluate_population(
    population: List[Individual],
    evaluator: Evaluator,
) -> int:
    """
    Evaluate fitness for all unevaluated individuals.
    Returns the number of evaluations performed.
    """
    evaluations = 0
    for individual in population:
        if individual.fitness == 0.0:
            individual.fitness = evaluator(individual.chunk_size, individual.top_k)
            evaluations += 1
    return evaluations


def tournament_selection(
    population: List[Individual],
    tournament_size: int = 3,
) -> Individual:
    """Select an individual using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def crossover(
    parent1: Individual,
    parent2: Individual,
    search_space: Dict[str, List[int]],
) -> Tuple[Individual, Individual]:
    """
    Perform uniform crossover between two parents.
    Each gene (parameter) has 50% chance of coming from either parent.
    """
    if random.random() < 0.5:
        child1_chunk = parent1.chunk_size
        child2_chunk = parent2.chunk_size
    else:
        child1_chunk = parent2.chunk_size
        child2_chunk = parent1.chunk_size

    if random.random() < 0.5:
        child1_top_k = parent1.top_k
        child2_top_k = parent2.top_k
    else:
        child1_top_k = parent2.top_k
        child2_top_k = parent1.top_k

    return (
        Individual(chunk_size=child1_chunk, top_k=child1_top_k),
        Individual(chunk_size=child2_chunk, top_k=child2_top_k),
    )


def mutate(
    individual: Individual,
    search_space: Dict[str, List[int]],
    mutation_rate: float = 0.2,
) -> Individual:
    """
    Mutate an individual by randomly changing parameters.
    Each parameter has mutation_rate probability of being mutated.
    """
    new_chunk_size = individual.chunk_size
    new_top_k = individual.top_k

    if random.random() < mutation_rate:
        chunk_sizes = search_space["chunk_size"]
        current_idx = chunk_sizes.index(individual.chunk_size)
        delta = random.choice([-1, 0, 1])
        new_idx = max(0, min(len(chunk_sizes) - 1, current_idx + delta))
        new_chunk_size = chunk_sizes[new_idx]

    if random.random() < mutation_rate:
        top_ks = search_space["top_k"]
        current_idx = top_ks.index(individual.top_k)
        delta = random.choice([-1, 0, 1])
        new_idx = max(0, min(len(top_ks) - 1, current_idx + delta))
        new_top_k = top_ks[new_idx]

    return Individual(chunk_size=new_chunk_size, top_k=new_top_k)


def genetic_algorithm(
    search_space: Dict[str, List[int]],
    max_evaluations: int,
    evaluator: Evaluator = evaluate_rag_pipeline,
    population_size: int = 6,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism_count: int = 1,
) -> Tuple[Dict[str, int], float]:
    """
    Performs genetic algorithm optimization to find the best RAG configuration.

    Args:
        search_space: The search space for the hyperparameters.
        max_evaluations: Maximum number of fitness evaluations to perform.
        evaluator: Function to evaluate a configuration.
        population_size: Number of individuals in the population.
        crossover_rate: Probability of performing crossover.
        mutation_rate: Probability of mutating each gene.
        elitism_count: Number of best individuals to preserve each generation.

    Returns:
        A tuple containing the best configuration found and its score.
    """
    population = [create_individual(search_space) for _ in range(population_size)]
    
    total_evaluations = evaluate_population(population, evaluator)
    
    best_individual = max(population, key=lambda x: x.fitness)
    best_config = best_individual.to_dict()
    best_score = best_individual.fitness

    generation = 0
    while total_evaluations < max_evaluations:
        generation += 1
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = population[:elitism_count]
        
        while len(new_population) < population_size:
            if total_evaluations >= max_evaluations:
                break
                
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, search_space)
            else:
                child1 = Individual(
                    chunk_size=parent1.chunk_size,
                    top_k=parent1.top_k,
                )
                child2 = Individual(
                    chunk_size=parent2.chunk_size,
                    top_k=parent2.top_k,
                )
            
            child1 = mutate(child1, search_space, mutation_rate)
            child2 = mutate(child2, search_space, mutation_rate)
            
            for child in [child1, child2]:
                if len(new_population) >= population_size:
                    break
                if total_evaluations >= max_evaluations:
                    break
                    
                child.fitness = evaluator(child.chunk_size, child.top_k)
                total_evaluations += 1
                new_population.append(child)
                
                if child.fitness > best_score:
                    best_score = child.fitness
                    best_config = child.to_dict()
        
        population = new_population

    return best_config, best_score
