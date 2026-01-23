"""Mutation operators for TSP genetic algorithm.

All mutations operate on permutation representations (arrays of city indices).
Optimized with Numba JIT compilation for performance on large instances.
"""

import numpy as np
from numba import njit
from typing import Callable


@njit(cache=True)
def swap_mutation(tour: np.ndarray) -> np.ndarray:
    """Swap two random cities in the tour.
    
    Classic mutation operator - simple but effective.
    Time complexity: O(1)
    
    Args:
        tour: Array of city indices representing a tour
        
    Returns:
        New tour with two cities swapped
    """
    n = len(tour)
    result = tour.copy()
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    result[i], result[j] = result[j], result[i]
    return result


@njit(cache=True)
def insert_mutation(tour: np.ndarray) -> np.ndarray:
    """Remove a city and insert it at a random position.
    
    More disruptive than swap - can help escape local optima.
    Time complexity: O(n)
    
    Args:
        tour: Array of city indices representing a tour
        
    Returns:
        New tour with one city relocated
    """
    n = len(tour)
    result = tour.copy()
    
    # Pick city to remove and new position
    remove_idx = np.random.randint(0, n)
    insert_idx = np.random.randint(0, n)
    
    if remove_idx == insert_idx:
        return result
    
    city = result[remove_idx]
    
    # Shift elements
    if remove_idx < insert_idx:
        for i in range(remove_idx, insert_idx):
            result[i] = result[i + 1]
        result[insert_idx] = city
    else:
        for i in range(remove_idx, insert_idx, -1):
            result[i] = result[i - 1]
        result[insert_idx] = city
    
    return result


@njit(cache=True)
def inversion_mutation(tour: np.ndarray) -> np.ndarray:
    """Reverse a random segment of the tour.
    
    Very powerful for TSP - equivalent to a 2-opt move.
    Helps eliminate edge crossings.
    Time complexity: O(n)
    
    Args:
        tour: Array of city indices representing a tour
        
    Returns:
        New tour with a segment reversed
    """
    n = len(tour)
    result = tour.copy()
    
    # Pick two cut points
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    if i > j:
        i, j = j, i
    
    # Reverse the segment between i and j
    while i < j:
        result[i], result[j] = result[j], result[i]
        i += 1
        j -= 1
    
    return result


@njit(cache=True)
def scramble_mutation(tour: np.ndarray) -> np.ndarray:
    """Randomly shuffle a segment of the tour.
    
    Highly disruptive - good for maintaining diversity.
    Time complexity: O(segment_length)
    
    Args:
        tour: Array of city indices representing a tour
        
    Returns:
        New tour with a segment scrambled
    """
    n = len(tour)
    result = tour.copy()
    
    # Pick segment bounds
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    if i > j:
        i, j = j, i
    
    # Fisher-Yates shuffle on the segment
    segment_len = j - i + 1
    for k in range(segment_len - 1, 0, -1):
        swap_idx = np.random.randint(0, k + 1)
        idx1, idx2 = i + k, i + swap_idx
        result[idx1], result[idx2] = result[idx2], result[idx1]
    
    return result


@njit(cache=True)
def displacement_mutation(tour: np.ndarray) -> np.ndarray:
    """Remove a segment and insert it elsewhere.
    
    Also known as "cut and paste" mutation.
    Combines aspects of insert and inversion.
    Time complexity: O(n)
    
    Args:
        tour: Array of city indices representing a tour
        
    Returns:
        New tour with a segment relocated
    """
    n = len(tour)
    if n < 4:
        return swap_mutation(tour)
    
    result = np.empty(n, dtype=tour.dtype)
    
    # Pick segment to move
    seg_start = np.random.randint(0, n - 1)
    seg_end = np.random.randint(seg_start + 1, min(seg_start + n // 3 + 2, n))
    seg_len = seg_end - seg_start
    
    # Pick insertion point (in the remaining sequence)
    remaining_len = n - seg_len
    if remaining_len == 0:
        return tour.copy()
    insert_pos = np.random.randint(0, remaining_len + 1)
    
    # Build new tour
    idx = 0
    remaining_idx = 0
    for i in range(n):
        if i < seg_start or i >= seg_end:
            if remaining_idx == insert_pos:
                # Insert the segment here
                for j in range(seg_start, seg_end):
                    result[idx] = tour[j]
                    idx += 1
            result[idx] = tour[i]
            idx += 1
            remaining_idx += 1
    
    # Handle case where segment goes at the end
    if remaining_idx == insert_pos:
        for j in range(seg_start, seg_end):
            result[idx] = tour[j]
            idx += 1
    
    return result


def apply_mutation(tour: np.ndarray, mutation_rate: float, 
                   mutation_func: Callable = None) -> np.ndarray:
    """Apply mutation with given probability.
    
    Args:
        tour: Array of city indices
        mutation_rate: Probability of mutation [0, 1]
        mutation_func: Mutation function to use (default: inversion)
        
    Returns:
        Possibly mutated tour
    """
    if mutation_func is None:
        mutation_func = inversion_mutation
    
    if np.random.random() < mutation_rate:
        return mutation_func(tour)
    return tour.copy()


def get_mutation_operator(name: str) -> Callable:
    """Get mutation operator by name.
    
    Args:
        name: One of 'swap', 'insert', 'inversion', 'scramble', 'displacement'
        
    Returns:
        Mutation function
    """
    operators = {
        'swap': swap_mutation,
        'insert': insert_mutation,
        'inversion': inversion_mutation,
        'scramble': scramble_mutation,
        'displacement': displacement_mutation,
    }
    if name not in operators:
        raise ValueError(f"Unknown mutation operator: {name}. "
                        f"Available: {list(operators.keys())}")
    return operators[name]


def adaptive_mutation(tour: np.ndarray, diversity: float, 
                      min_rate: float = 0.01, max_rate: float = 0.3) -> np.ndarray:
    """Apply mutation with rate adapted to population diversity.
    
    When diversity is low, increase mutation rate to escape local optima.
    When diversity is high, decrease rate to allow convergence.
    
    Args:
        tour: Array of city indices
        diversity: Current population diversity metric [0, 1]
        min_rate: Minimum mutation rate
        max_rate: Maximum mutation rate
        
    Returns:
        Possibly mutated tour
    """
    # Inverse relationship: low diversity -> high mutation
    rate = max_rate - diversity * (max_rate - min_rate)
    rate = np.clip(rate, min_rate, max_rate)
    
    # Use inversion as default (best for TSP)
    if np.random.random() < rate:
        return inversion_mutation(tour)
    return tour.copy()


def compound_mutation(tour: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Apply a random mutation operator.
    
    Randomly selects from available operators for diversity.
    
    Args:
        tour: Array of city indices
        mutation_rate: Probability of mutation
        
    Returns:
        Possibly mutated tour
    """
    if np.random.random() >= mutation_rate:
        return tour.copy()
    
    # Weighted selection favoring inversion (best for TSP)
    operators = [
        (inversion_mutation, 0.4),
        (swap_mutation, 0.2),
        (insert_mutation, 0.2),
        (scramble_mutation, 0.1),
        (displacement_mutation, 0.1),
    ]
    
    r = np.random.random()
    cumsum = 0.0
    for op, weight in operators:
        cumsum += weight
        if r < cumsum:
            return op(tour)
    
    return inversion_mutation(tour)
