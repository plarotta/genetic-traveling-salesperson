"""Crossover operators for TSP genetic algorithm.

All crossover operators produce valid permutations (no duplicates).
Optimized with Numba JIT compilation for performance on large instances.
"""

import numpy as np
from numba import njit
from typing import Tuple, Callable


@njit(cache=True)
def order_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order Crossover (OX) - preserves relative ordering.
    
    One of the best crossover operators for TSP.
    Copies a segment from parent1, fills remaining positions with
    cities from parent2 in order (skipping those already present).
    
    Time complexity: O(n)
    
    Args:
        parent1: First parent tour (city indices)
        parent2: Second parent tour (city indices)
        
    Returns:
        Two offspring tours
    """
    n = len(parent1)
    
    # Pick two crossover points
    cp1, cp2 = np.random.randint(0, n), np.random.randint(0, n)
    if cp1 > cp2:
        cp1, cp2 = cp2, cp1
    
    # Create offspring
    child1 = np.full(n, -1, dtype=parent1.dtype)
    child2 = np.full(n, -1, dtype=parent2.dtype)
    
    # Copy segment from parents
    for i in range(cp1, cp2 + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    
    # Track which cities are in each child
    in_child1 = np.zeros(n, dtype=np.bool_)
    in_child2 = np.zeros(n, dtype=np.bool_)
    for i in range(cp1, cp2 + 1):
        in_child1[parent1[i]] = True
        in_child2[parent2[i]] = True
    
    # Fill child1 with cities from parent2
    fill_idx = (cp2 + 1) % n
    for i in range(n):
        city = parent2[(cp2 + 1 + i) % n]
        if not in_child1[city]:
            child1[fill_idx] = city
            fill_idx = (fill_idx + 1) % n
            if fill_idx == cp1:
                fill_idx = (cp2 + 1) % n
    
    # Fill child2 with cities from parent1
    fill_idx = (cp2 + 1) % n
    for i in range(n):
        city = parent1[(cp2 + 1 + i) % n]
        if not in_child2[city]:
            child2[fill_idx] = city
            fill_idx = (fill_idx + 1) % n
            if fill_idx == cp1:
                fill_idx = (cp2 + 1) % n
    
    return child1, child2


@njit(cache=True)
def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Partially Mapped Crossover (PMX) - preserves absolute positions.
    
    Creates mapping between segments and uses it to fill positions.
    Good at preserving position information from both parents.
    
    Time complexity: O(n)
    
    Args:
        parent1: First parent tour (city indices)
        parent2: Second parent tour (city indices)
        
    Returns:
        Two offspring tours
    """
    n = len(parent1)
    
    # Pick two crossover points
    cp1, cp2 = np.random.randint(0, n), np.random.randint(0, n)
    if cp1 > cp2:
        cp1, cp2 = cp2, cp1
    
    # Initialize children
    child1 = np.full(n, -1, dtype=parent1.dtype)
    child2 = np.full(n, -1, dtype=parent2.dtype)
    
    # Copy segments
    for i in range(cp1, cp2 + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    
    # Create position lookup for parents
    pos1 = np.empty(n, dtype=np.int64)
    pos2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos1[parent1[i]] = i
        pos2[parent2[i]] = i
    
    # Track used cities
    used1 = np.zeros(n, dtype=np.bool_)
    used2 = np.zeros(n, dtype=np.bool_)
    for i in range(cp1, cp2 + 1):
        used1[child1[i]] = True
        used2[child2[i]] = True
    
    # Fill remaining positions for child1 using mapping
    for i in range(n):
        if i >= cp1 and i <= cp2:
            continue
        city = parent2[i]
        while used1[city]:
            # Find where this city is in parent1's segment
            idx = pos1[city]
            if idx >= cp1 and idx <= cp2:
                city = parent2[idx]
            else:
                break
        child1[i] = city
        used1[city] = True
    
    # Fill remaining positions for child2 using mapping
    for i in range(n):
        if i >= cp1 and i <= cp2:
            continue
        city = parent1[i]
        while used2[city]:
            idx = pos2[city]
            if idx >= cp1 and idx <= cp2:
                city = parent1[idx]
            else:
                break
        child2[i] = city
        used2[city] = True
    
    return child1, child2


@njit(cache=True)
def cycle_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cycle Crossover (CX) - preserves absolute positions from both parents.
    
    Identifies cycles of positions and alternates taking from each parent.
    Each position in offspring comes from the same position in one parent.
    
    Time complexity: O(n)
    
    Args:
        parent1: First parent tour (city indices)
        parent2: Second parent tour (city indices)
        
    Returns:
        Two offspring tours
    """
    n = len(parent1)
    
    child1 = np.empty(n, dtype=parent1.dtype)
    child2 = np.empty(n, dtype=parent2.dtype)
    
    # Create position lookup for parent1
    pos1 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos1[parent1[i]] = i
    
    # Track which positions have been filled
    filled = np.zeros(n, dtype=np.bool_)
    
    cycle_num = 0
    start = 0
    
    while start < n:
        # Find start of next unfilled cycle
        while start < n and filled[start]:
            start += 1
        if start >= n:
            break
        
        # Trace the cycle
        idx = start
        while not filled[idx]:
            filled[idx] = True
            if cycle_num % 2 == 0:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
            else:
                child1[idx] = parent2[idx]
                child2[idx] = parent1[idx]
            # Move to position of this element in other parent
            idx = pos1[parent2[idx]]
        
        cycle_num += 1
    
    return child1, child2


@njit(cache=True)
def edge_recombination_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Edge Recombination Crossover (ERX) - preserves edges from both parents.
    
    Builds an edge table showing neighbors of each city in both parents.
    Constructs offspring by preferring cities with fewer remaining edges.
    
    One of the best operators for TSP as it preserves edge information.
    
    Time complexity: O(n^2) worst case, typically O(n)
    
    Args:
        parent1: First parent tour (city indices)  
        parent2: Second parent tour (city indices)
        
    Returns:
        Two offspring tours
    """
    n = len(parent1)
    
    # Build edge table (adjacency list for each city)
    # Each city can have up to 4 neighbors (2 from each parent)
    edge_table = np.full((n, 4), -1, dtype=np.int64)
    edge_count = np.zeros(n, dtype=np.int64)
    
    # Add edges from parent1
    for i in range(n):
        city = parent1[i]
        left = parent1[(i - 1) % n]
        right = parent1[(i + 1) % n]
        
        edge_table[city, edge_count[city]] = left
        edge_count[city] += 1
        edge_table[city, edge_count[city]] = right
        edge_count[city] += 1
    
    # Add edges from parent2 (avoiding duplicates)
    for i in range(n):
        city = parent2[i]
        left = parent2[(i - 1) % n]
        right = parent2[(i + 1) % n]
        
        for neighbor in [left, right]:
            # Check if already present
            found = False
            for j in range(edge_count[city]):
                if edge_table[city, j] == neighbor:
                    found = True
                    break
            if not found:
                edge_table[city, edge_count[city]] = neighbor
                edge_count[city] += 1
    
    # Build first child
    child1 = np.empty(n, dtype=parent1.dtype)
    used = np.zeros(n, dtype=np.bool_)
    
    # Start with first city of parent1
    current = parent1[0]
    child1[0] = current
    used[current] = True
    
    for pos in range(1, n):
        # Remove current city from all edge lists
        for city in range(n):
            for j in range(edge_count[city]):
                if edge_table[city, j] == current:
                    # Shift remaining edges
                    for k in range(j, edge_count[city] - 1):
                        edge_table[city, k] = edge_table[city, k + 1]
                    edge_count[city] -= 1
                    break
        
        # Find next city: prefer neighbors with fewest edges
        best_neighbor = -1
        best_count = 5  # More than max possible
        
        for j in range(4):
            neighbor = edge_table[current, j]
            if neighbor >= 0 and not used[neighbor]:
                if edge_count[neighbor] < best_count:
                    best_count = edge_count[neighbor]
                    best_neighbor = neighbor
        
        if best_neighbor == -1:
            # No valid neighbor, pick random unused city
            for city in range(n):
                if not used[city]:
                    best_neighbor = city
                    break
        
        current = best_neighbor
        child1[pos] = current
        used[current] = True
    
    # Build second child similarly but start from parent2
    # Rebuild edge table
    edge_count[:] = 0
    edge_table[:, :] = -1
    
    for i in range(n):
        city = parent1[i]
        left = parent1[(i - 1) % n]
        right = parent1[(i + 1) % n]
        edge_table[city, edge_count[city]] = left
        edge_count[city] += 1
        edge_table[city, edge_count[city]] = right
        edge_count[city] += 1
    
    for i in range(n):
        city = parent2[i]
        left = parent2[(i - 1) % n]
        right = parent2[(i + 1) % n]
        for neighbor in [left, right]:
            found = False
            for j in range(edge_count[city]):
                if edge_table[city, j] == neighbor:
                    found = True
                    break
            if not found:
                edge_table[city, edge_count[city]] = neighbor
                edge_count[city] += 1
    
    child2 = np.empty(n, dtype=parent2.dtype)
    used[:] = False
    
    current = parent2[0]
    child2[0] = current
    used[current] = True
    
    for pos in range(1, n):
        for city in range(n):
            for j in range(edge_count[city]):
                if edge_table[city, j] == current:
                    for k in range(j, edge_count[city] - 1):
                        edge_table[city, k] = edge_table[city, k + 1]
                    edge_count[city] -= 1
                    break
        
        best_neighbor = -1
        best_count = 5
        
        for j in range(4):
            neighbor = edge_table[current, j]
            if neighbor >= 0 and not used[neighbor]:
                if edge_count[neighbor] < best_count:
                    best_count = edge_count[neighbor]
                    best_neighbor = neighbor
        
        if best_neighbor == -1:
            for city in range(n):
                if not used[city]:
                    best_neighbor = city
                    break
        
        current = best_neighbor
        child2[pos] = current
        used[current] = True
    
    return child1, child2


def apply_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                    crossover_rate: float,
                    crossover_func: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply crossover with given probability.
    
    Args:
        parent1: First parent tour
        parent2: Second parent tour
        crossover_rate: Probability of crossover [0, 1]
        crossover_func: Crossover function to use (default: order_crossover)
        
    Returns:
        Two offspring (may be copies of parents if no crossover)
    """
    if crossover_func is None:
        crossover_func = order_crossover
    
    if np.random.random() < crossover_rate:
        return crossover_func(parent1, parent2)
    return parent1.copy(), parent2.copy()


def get_crossover_operator(name: str) -> Callable:
    """Get crossover operator by name.
    
    Args:
        name: One of 'ox', 'pmx', 'cx', 'erx'
        
    Returns:
        Crossover function
    """
    operators = {
        'ox': order_crossover,
        'order': order_crossover,
        'pmx': pmx_crossover,
        'partially_mapped': pmx_crossover,
        'cx': cycle_crossover,
        'cycle': cycle_crossover,
        'erx': edge_recombination_crossover,
        'edge_recombination': edge_recombination_crossover,
    }
    if name.lower() not in operators:
        raise ValueError(f"Unknown crossover operator: {name}. "
                        f"Available: {list(operators.keys())}")
    return operators[name.lower()]
