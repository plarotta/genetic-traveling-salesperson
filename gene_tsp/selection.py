"""Selection operators for genetic algorithms.

Includes tournament selection, elitism, and diversity-preserving mechanisms.
"""

import numpy as np
from numba import njit
from typing import Tuple, List


@njit(cache=True)
def tournament_selection(fitness: np.ndarray, tournament_size: int = 3) -> int:
    """Select individual using tournament selection.
    
    More scalable than roulette wheel and provides good selection pressure.
    Lower fitness is better (minimization).
    
    Args:
        fitness: Array of fitness values for each individual
        tournament_size: Number of individuals in tournament
        
    Returns:
        Index of selected individual
    """
    n = len(fitness)
    best_idx = np.random.randint(0, n)
    best_fitness = fitness[best_idx]
    
    for _ in range(tournament_size - 1):
        idx = np.random.randint(0, n)
        if fitness[idx] < best_fitness:  # Lower is better
            best_idx = idx
            best_fitness = fitness[idx]
    
    return best_idx


@njit(cache=True)
def tournament_selection_batch(fitness: np.ndarray, n_selections: int, 
                                tournament_size: int = 3) -> np.ndarray:
    """Perform multiple tournament selections.
    
    Args:
        fitness: Array of fitness values
        n_selections: Number of individuals to select
        tournament_size: Size of each tournament
        
    Returns:
        Array of selected indices
    """
    selected = np.empty(n_selections, dtype=np.int64)
    for i in range(n_selections):
        selected[i] = tournament_selection(fitness, tournament_size)
    return selected


@njit(cache=True)
def roulette_wheel_selection(fitness: np.ndarray) -> int:
    """Roulette wheel selection (fitness proportionate).
    
    For minimization: transforms fitness to selection probability.
    Lower fitness values get higher selection probability.
    
    Args:
        fitness: Array of fitness values (lower is better)
        
    Returns:
        Index of selected individual
    """
    n = len(fitness)
    
    # Transform to maximization (invert)
    max_fit = np.max(fitness)
    transformed = max_fit - fitness + 1e-6  # Add small constant to avoid zero
    
    # Normalize to probabilities
    total = np.sum(transformed)
    probs = transformed / total
    
    # Spin the wheel
    r = np.random.random()
    cumsum = 0.0
    for i in range(n):
        cumsum += probs[i]
        if r <= cumsum:
            return i
    
    return n - 1


@njit(cache=True)
def rank_selection(fitness: np.ndarray) -> int:
    """Rank-based selection.
    
    Selection probability based on rank rather than raw fitness.
    More robust to fitness scaling issues.
    
    Args:
        fitness: Array of fitness values (lower is better)
        
    Returns:
        Index of selected individual
    """
    n = len(fitness)
    
    # Get ranks (1 = best, n = worst)
    indices = np.argsort(fitness)
    ranks = np.empty(n, dtype=np.float64)
    for i, idx in enumerate(indices):
        ranks[idx] = n - i  # Higher rank for lower fitness
    
    # Selection probability proportional to rank
    total = np.sum(ranks)
    r = np.random.random() * total
    cumsum = 0.0
    for i in range(n):
        cumsum += ranks[i]
        if r <= cumsum:
            return i
    
    return n - 1


def get_elite(population: np.ndarray, fitness: np.ndarray, 
              n_elite: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get the elite individuals from population.
    
    Args:
        population: Array of shape (pop_size, n_cities)
        fitness: Array of fitness values
        n_elite: Number of elite individuals to preserve
        
    Returns:
        Tuple of (elite_population, elite_fitness)
    """
    indices = np.argsort(fitness)[:n_elite]
    return population[indices].copy(), fitness[indices].copy()


def select_parents(population: np.ndarray, fitness: np.ndarray,
                   n_parents: int, method: str = 'tournament',
                   tournament_size: int = 3) -> np.ndarray:
    """Select parents for reproduction.
    
    Args:
        population: Array of shape (pop_size, n_cities)
        fitness: Array of fitness values
        n_parents: Number of parents to select
        method: Selection method ('tournament', 'roulette', 'rank')
        tournament_size: Size of tournament (if applicable)
        
    Returns:
        Array of selected parent indices
    """
    if method == 'tournament':
        return tournament_selection_batch(fitness, n_parents, tournament_size)
    elif method == 'roulette':
        return np.array([roulette_wheel_selection(fitness) 
                        for _ in range(n_parents)])
    elif method == 'rank':
        return np.array([rank_selection(fitness) for _ in range(n_parents)])
    else:
        raise ValueError(f"Unknown selection method: {method}")


@njit(cache=True)
def crowding_replacement(population: np.ndarray, fitness: np.ndarray,
                         offspring: np.ndarray, offspring_fitness: np.ndarray,
                         distance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace individuals using crowding (diversity preservation).
    
    Each offspring replaces the most similar individual in a random subset.
    Helps maintain population diversity.
    
    Args:
        population: Current population (pop_size, n_cities)
        fitness: Current fitness values
        offspring: New offspring (n_offspring, n_cities)
        offspring_fitness: Offspring fitness values
        distance_matrix: Precomputed distance matrix
        
    Returns:
        Updated (population, fitness)
    """
    pop_size = len(population)
    n_offspring = len(offspring)
    
    new_pop = population.copy()
    new_fit = fitness.copy()
    
    crowd_size = min(3, pop_size)
    
    for i in range(n_offspring):
        # Select random subset
        subset = np.random.choice(pop_size, crowd_size, replace=False)
        
        # Find most similar (based on genotype distance)
        best_match = subset[0]
        best_dist = 1e10
        
        for idx in subset:
            # Simple genotype distance: number of different positions
            dist = 0.0
            for j in range(len(offspring[i])):
                if offspring[i, j] != new_pop[idx, j]:
                    dist += 1
            if dist < best_dist:
                best_dist = dist
                best_match = idx
        
        # Replace if offspring is better
        if offspring_fitness[i] < new_fit[best_match]:
            new_pop[best_match] = offspring[i]
            new_fit[best_match] = offspring_fitness[i]
    
    return new_pop, new_fit


@njit(cache=True)
def calculate_diversity(population: np.ndarray) -> float:
    """Calculate population diversity as average pairwise distance.
    
    Uses Hamming distance between tours (number of differing positions).
    
    Args:
        population: Array of shape (pop_size, n_cities)
        
    Returns:
        Diversity metric in [0, 1]
    """
    pop_size = len(population)
    n_cities = len(population[0])
    
    if pop_size < 2:
        return 0.0
    
    total_dist = 0.0
    n_pairs = 0
    
    # Sample pairs for efficiency
    max_pairs = min(100, pop_size * (pop_size - 1) // 2)
    
    for _ in range(max_pairs):
        i = np.random.randint(0, pop_size)
        j = np.random.randint(0, pop_size)
        if i != j:
            dist = 0
            for k in range(n_cities):
                if population[i, k] != population[j, k]:
                    dist += 1
            total_dist += dist
            n_pairs += 1
    
    if n_pairs == 0:
        return 0.0
    
    # Normalize by max possible distance
    avg_dist = total_dist / n_pairs
    return avg_dist / n_cities


def survival_selection(population: np.ndarray, fitness: np.ndarray,
                       offspring: np.ndarray, offspring_fitness: np.ndarray,
                       pop_size: int, n_elite: int = 2,
                       method: str = 'elitist') -> Tuple[np.ndarray, np.ndarray]:
    """Select survivors for next generation.
    
    Args:
        population: Current population
        fitness: Current fitness values
        offspring: New offspring
        offspring_fitness: Offspring fitness values
        pop_size: Target population size
        n_elite: Number of elite to always preserve
        method: 'elitist' or 'generational'
        
    Returns:
        Tuple of (new_population, new_fitness)
    """
    if method == 'elitist':
        # Combine parents and offspring
        combined_pop = np.vstack([population, offspring])
        combined_fit = np.concatenate([fitness, offspring_fitness])
        
        # Get elite
        elite_pop, elite_fit = get_elite(combined_pop, combined_fit, n_elite)
        
        # Tournament selection for remaining slots
        remaining = pop_size - n_elite
        indices = tournament_selection_batch(combined_fit, remaining, 3)
        
        new_pop = np.vstack([elite_pop, combined_pop[indices]])
        new_fit = np.concatenate([elite_fit, combined_fit[indices]])
        
        return new_pop, new_fit
    
    elif method == 'generational':
        # Keep only elite from parents, rest from offspring
        elite_pop, elite_fit = get_elite(population, fitness, n_elite)
        
        remaining = pop_size - n_elite
        if len(offspring) >= remaining:
            indices = np.argsort(offspring_fitness)[:remaining]
            new_pop = np.vstack([elite_pop, offspring[indices]])
            new_fit = np.concatenate([elite_fit, offspring_fitness[indices]])
        else:
            new_pop = np.vstack([elite_pop, offspring])
            new_fit = np.concatenate([elite_fit, offspring_fitness])
        
        return new_pop, new_fit
    
    else:
        raise ValueError(f"Unknown survival selection method: {method}")
