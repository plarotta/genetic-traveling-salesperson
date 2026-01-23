"""Local search operators for TSP.

Includes 2-opt, 3-opt, and Or-opt algorithms with Numba optimization.
These can dramatically improve solution quality when applied to GA offspring.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional


@njit(cache=True)
def two_opt(tour: np.ndarray, dist_matrix: np.ndarray, 
            max_iterations: int = -1) -> Tuple[np.ndarray, float]:
    """2-opt local search - reverses segments to eliminate crossings.
    
    This is the most important improvement for TSP genetic algorithms.
    Iteratively finds and applies improving 2-opt moves.
    
    Time complexity: O(n^2) per iteration, typically O(n^2 * k) total
    where k is a small constant.
    
    Args:
        tour: Initial tour (city indices)
        dist_matrix: Precomputed distance matrix
        max_iterations: Maximum iterations (-1 for unlimited)
        
    Returns:
        Tuple of (improved_tour, final_length)
    """
    n = len(tour)
    best_tour = tour.copy()
    improved = True
    iterations = 0
    
    while improved:
        if max_iterations > 0 and iterations >= max_iterations:
            break
        improved = False
        iterations += 1
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == i + 1:
                    continue
                # Skip if j wraps around to i
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate improvement
                # Current edges: (i, i+1) and (j, j+1)
                # New edges: (i, j) and (i+1, j+1)
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                
                old_dist = dist_matrix[a, b] + dist_matrix[c, d]
                new_dist = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_dist < old_dist - 1e-10:
                    # Apply 2-opt move: reverse segment from i+1 to j
                    left, right = i + 1, j
                    while left < right:
                        best_tour[left], best_tour[right] = best_tour[right], best_tour[left]
                        left += 1
                        right -= 1
                    improved = True
    
    # Calculate final length
    length = 0.0
    for i in range(n):
        length += dist_matrix[best_tour[i], best_tour[(i + 1) % n]]
    
    return best_tour, length


@njit(cache=True)
def two_opt_iteration(tour: np.ndarray, dist_matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Single iteration of 2-opt (for use in parallel processing).
    
    Performs one complete scan looking for improvements.
    
    Args:
        tour: Current tour
        dist_matrix: Precomputed distance matrix
        
    Returns:
        Tuple of (possibly_improved_tour, was_improved)
    """
    n = len(tour)
    best_tour = tour.copy()
    best_delta = 0.0
    best_i, best_j = -1, -1
    
    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            
            a, b = best_tour[i], best_tour[i + 1]
            c, d = best_tour[j], best_tour[(j + 1) % n]
            
            delta = (dist_matrix[a, c] + dist_matrix[b, d] - 
                    dist_matrix[a, b] - dist_matrix[c, d])
            
            if delta < best_delta - 1e-10:
                best_delta = delta
                best_i, best_j = i, j
    
    if best_i >= 0:
        left, right = best_i + 1, best_j
        while left < right:
            best_tour[left], best_tour[right] = best_tour[right], best_tour[left]
            left += 1
            right -= 1
        return best_tour, True
    
    return best_tour, False


@njit(cache=True)
def three_opt(tour: np.ndarray, dist_matrix: np.ndarray,
              max_iterations: int = 10) -> Tuple[np.ndarray, float]:
    """3-opt local search - more powerful than 2-opt but slower.
    
    Considers breaking the tour into 3 segments and reconnecting them
    in different ways. Can escape local optima that 2-opt cannot.
    
    Time complexity: O(n^3) per iteration - use sparingly on large instances.
    
    Args:
        tour: Initial tour (city indices)
        dist_matrix: Precomputed distance matrix
        max_iterations: Maximum iterations
        
    Returns:
        Tuple of (improved_tour, final_length)
    """
    n = len(tour)
    best_tour = tour.copy()
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    # Skip if segments are too small
                    if k == n - 1 and i == 0:
                        continue
                    
                    # Current tour length for these edges
                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[j + 1]
                    e, f = best_tour[k], best_tour[(k + 1) % n]
                    
                    current = (dist_matrix[a, b] + dist_matrix[c, d] + 
                              dist_matrix[e, f])
                    
                    # Try different reconnection options
                    # Option 1: Reverse segment i+1 to j
                    opt1 = (dist_matrix[a, c] + dist_matrix[b, d] + 
                           dist_matrix[e, f])
                    
                    # Option 2: Reverse segment j+1 to k
                    opt2 = (dist_matrix[a, b] + dist_matrix[c, e] + 
                           dist_matrix[d, f])
                    
                    # Option 3: Reverse both segments
                    opt3 = (dist_matrix[a, c] + dist_matrix[b, e] + 
                           dist_matrix[d, f])
                    
                    # Option 4: Alternative reconnection
                    opt4 = (dist_matrix[a, d] + dist_matrix[e, b] + 
                           dist_matrix[c, f])
                    
                    best_option = current
                    best_idx = 0
                    
                    for idx, opt in enumerate([opt1, opt2, opt3, opt4], 1):
                        if opt < best_option - 1e-10:
                            best_option = opt
                            best_idx = idx
                    
                    if best_idx > 0:
                        # Apply the best move
                        new_tour = best_tour.copy()
                        
                        if best_idx == 1:
                            # Reverse segment i+1 to j
                            left, right = i + 1, j
                            while left < right:
                                new_tour[left], new_tour[right] = new_tour[right], new_tour[left]
                                left += 1
                                right -= 1
                        elif best_idx == 2:
                            # Reverse segment j+1 to k
                            left, right = j + 1, k
                            while left < right:
                                new_tour[left], new_tour[right] = new_tour[right], new_tour[left]
                                left += 1
                                right -= 1
                        elif best_idx == 3:
                            # Reverse both segments
                            left, right = i + 1, j
                            while left < right:
                                new_tour[left], new_tour[right] = new_tour[right], new_tour[left]
                                left += 1
                                right -= 1
                            left, right = j + 1, k
                            while left < right:
                                new_tour[left], new_tour[right] = new_tour[right], new_tour[left]
                                left += 1
                                right -= 1
                        elif best_idx == 4:
                            # More complex reconnection - rebuild
                            temp = np.empty(n, dtype=best_tour.dtype)
                            idx = 0
                            for x in range(0, i + 1):
                                temp[idx] = best_tour[x]
                                idx += 1
                            for x in range(j + 1, k + 1):
                                temp[idx] = best_tour[x]
                                idx += 1
                            for x in range(i + 1, j + 1):
                                temp[idx] = best_tour[x]
                                idx += 1
                            for x in range(k + 1, n):
                                temp[idx] = best_tour[x]
                                idx += 1
                            new_tour = temp
                        
                        best_tour = new_tour
                        improved = True
    
    # Calculate final length
    length = 0.0
    for i in range(n):
        length += dist_matrix[best_tour[i], best_tour[(i + 1) % n]]
    
    return best_tour, length


@njit(cache=True)
def or_opt(tour: np.ndarray, dist_matrix: np.ndarray,
           max_segment_size: int = 3) -> Tuple[np.ndarray, float]:
    """Or-opt local search - relocates chains of cities.
    
    Moves segments of 1, 2, or 3 consecutive cities to other positions.
    Less expensive than 3-opt but can find similar improvements.
    
    Time complexity: O(n^2) per iteration
    
    Args:
        tour: Initial tour (city indices)
        dist_matrix: Precomputed distance matrix
        max_segment_size: Maximum segment length to relocate (1-3)
        
    Returns:
        Tuple of (improved_tour, final_length)
    """
    n = len(tour)
    best_tour = tour.copy()
    improved = True
    
    while improved:
        improved = False
        
        for seg_size in range(1, min(max_segment_size + 1, n - 2)):
            for i in range(n):
                # Segment to move: tour[i:i+seg_size]
                seg_end = (i + seg_size - 1) % n
                
                # Current cost of having segment here
                before_seg = (i - 1) % n
                after_seg = (i + seg_size) % n
                
                if before_seg == seg_end or after_seg == i:
                    continue
                
                a = best_tour[before_seg]  # Before segment
                b = best_tour[i]           # Start of segment
                c = best_tour[seg_end]     # End of segment
                d = best_tour[after_seg]   # After segment
                
                current_cost = (dist_matrix[a, b] + dist_matrix[c, d])
                removal_saving = current_cost - dist_matrix[a, d]
                
                # Try inserting segment at each other position
                for j in range(n):
                    # Skip positions in or adjacent to current segment location
                    if j >= before_seg and j <= (after_seg % n):
                        continue
                    if abs(j - i) <= seg_size:
                        continue
                    
                    j_next = (j + 1) % n
                    
                    # Cost of inserting segment between j and j_next
                    e = best_tour[j]
                    f = best_tour[j_next]
                    
                    insertion_cost = (dist_matrix[e, b] + dist_matrix[c, f] - 
                                     dist_matrix[e, f])
                    
                    delta = insertion_cost - removal_saving
                    
                    if delta < -1e-10:
                        # Apply the move
                        new_tour = np.empty(n, dtype=best_tour.dtype)
                        
                        # This is complex - rebuild tour
                        # Remove segment from position i, insert after position j
                        segment = np.empty(seg_size, dtype=best_tour.dtype)
                        for k in range(seg_size):
                            segment[k] = best_tour[(i + k) % n]
                        
                        # Build new tour
                        idx = 0
                        pos = 0
                        while idx < n:
                            # Skip the original segment positions
                            skip = False
                            for k in range(seg_size):
                                if pos == (i + k) % n:
                                    skip = True
                                    break
                            
                            if skip:
                                pos = (pos + 1) % n
                                continue
                            
                            new_tour[idx] = best_tour[pos]
                            idx += 1
                            
                            # Insert segment after adjusted position j
                            if pos == j and idx <= n - seg_size:
                                for k in range(seg_size):
                                    if idx < n:
                                        new_tour[idx] = segment[k]
                                        idx += 1
                            
                            pos = (pos + 1) % n
                        
                        best_tour = new_tour
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    # Calculate final length
    length = 0.0
    for i in range(n):
        length += dist_matrix[best_tour[i], best_tour[(i + 1) % n]]
    
    return best_tour, length


@njit(cache=True)
def lin_kernighan_light(tour: np.ndarray, dist_matrix: np.ndarray,
                        max_depth: int = 5) -> Tuple[np.ndarray, float]:
    """Simplified Lin-Kernighan style improvement.
    
    Performs variable-depth search using 2-opt moves as building blocks.
    Less powerful than full LK but faster and still effective.
    
    Args:
        tour: Initial tour
        dist_matrix: Precomputed distance matrix
        max_depth: Maximum search depth
        
    Returns:
        Tuple of (improved_tour, final_length)
    """
    n = len(tour)
    best_tour = tour.copy()
    
    # Start with 2-opt to get a good baseline
    best_tour, _ = two_opt(best_tour, dist_matrix, max_iterations=3)
    
    improved = True
    while improved:
        improved = False
        
        for start_i in range(n):
            # Try a sequence of moves starting from this edge
            current_tour = best_tour.copy()
            total_gain = 0.0
            
            for depth in range(max_depth):
                best_gain = 0.0
                best_j = -1
                
                # Find best 2-opt move from current position
                i = (start_i + depth) % n
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    
                    a, b = current_tour[i], current_tour[i + 1]
                    c, d = current_tour[j], current_tour[(j + 1) % n]
                    
                    gain = (dist_matrix[a, b] + dist_matrix[c, d] -
                           dist_matrix[a, c] - dist_matrix[b, d])
                    
                    if gain > best_gain + 1e-10:
                        best_gain = gain
                        best_j = j
                
                if best_j >= 0:
                    # Apply the move
                    left, right = i + 1, best_j
                    while left < right:
                        current_tour[left], current_tour[right] = \
                            current_tour[right], current_tour[left]
                        left += 1
                        right -= 1
                    total_gain += best_gain
                else:
                    break
            
            if total_gain > 1e-10:
                best_tour = current_tour
                improved = True
    
    # Calculate final length
    length = 0.0
    for i in range(n):
        length += dist_matrix[best_tour[i], best_tour[(i + 1) % n]]
    
    return best_tour, length


def local_search(tour: np.ndarray, dist_matrix: np.ndarray,
                 method: str = '2opt', **kwargs) -> Tuple[np.ndarray, float]:
    """Apply local search to improve a tour.
    
    Args:
        tour: Initial tour (city indices)
        dist_matrix: Precomputed distance matrix
        method: '2opt', '3opt', 'or_opt', or 'lk'
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Tuple of (improved_tour, final_length)
    """
    if method == '2opt':
        return two_opt(tour, dist_matrix, **kwargs)
    elif method == '3opt':
        return three_opt(tour, dist_matrix, **kwargs)
    elif method == 'or_opt':
        return or_opt(tour, dist_matrix, **kwargs)
    elif method == 'lk':
        return lin_kernighan_light(tour, dist_matrix, **kwargs)
    else:
        raise ValueError(f"Unknown local search method: {method}")


@njit(cache=True, parallel=True)
def batch_two_opt(tours: np.ndarray, dist_matrix: np.ndarray,
                  max_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 2-opt to multiple tours in parallel.
    
    Args:
        tours: Array of shape (n_tours, n_cities)
        dist_matrix: Precomputed distance matrix
        max_iterations: Max iterations per tour
        
    Returns:
        Tuple of (improved_tours, lengths)
    """
    n_tours = len(tours)
    improved = np.empty_like(tours)
    lengths = np.empty(n_tours, dtype=np.float64)
    
    for i in prange(n_tours):
        improved[i], lengths[i] = two_opt(tours[i], dist_matrix, max_iterations)
    
    return improved, lengths
