"""Distance matrix computation and optimized fitness evaluation.

Precomputes distances for O(1) lookup during fitness evaluation.
Supports both CPU (NumPy/Numba) and GPU (CuPy) backends.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, Tuple
import warnings


class DistanceMatrix:
    """Precomputed distance matrix for efficient fitness evaluation.
    
    Attributes:
        coords: Original city coordinates (n_cities, 2)
        matrix: Distance matrix (n_cities, n_cities)
        n_cities: Number of cities
        use_gpu: Whether GPU acceleration is enabled
    """
    
    def __init__(self, coords: np.ndarray, use_gpu: bool = False):
        """Initialize distance matrix.
        
        Args:
            coords: City coordinates of shape (n_cities, 2)
            use_gpu: Attempt to use GPU acceleration if available
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_cities = len(coords)
        self.use_gpu = False
        self._gpu_matrix = None
        
        # Compute distance matrix
        if use_gpu:
            try:
                self.matrix = self._compute_gpu()
                self.use_gpu = True
            except ImportError:
                warnings.warn("CuPy not available, falling back to CPU")
                self.matrix = self._compute_cpu()
        else:
            self.matrix = self._compute_cpu()
        
        # Precompute nearest neighbors for each city
        self.nearest_neighbors = self._compute_nearest_neighbors()
    
    def _compute_cpu(self) -> np.ndarray:
        """Compute distance matrix on CPU."""
        return compute_distance_matrix(self.coords)
    
    def _compute_gpu(self) -> np.ndarray:
        """Compute distance matrix on GPU using CuPy."""
        import cupy as cp
        
        coords_gpu = cp.asarray(self.coords)
        
        # Compute pairwise distances using broadcasting
        diff = coords_gpu[:, np.newaxis, :] - coords_gpu[np.newaxis, :, :]
        dist = cp.sqrt(cp.sum(diff ** 2, axis=2))
        
        self._gpu_matrix = dist
        return cp.asnumpy(dist)
    
    def _compute_nearest_neighbors(self, k: int = 20) -> np.ndarray:
        """Compute k nearest neighbors for each city.
        
        Useful for constructive heuristics and local search.
        
        Args:
            k: Number of nearest neighbors to store
            
        Returns:
            Array of shape (n_cities, k) with neighbor indices
        """
        k = min(k, self.n_cities - 1)
        nn = np.empty((self.n_cities, k), dtype=np.int64)
        
        for i in range(self.n_cities):
            # Get distances from city i, excluding self
            distances = self.matrix[i].copy()
            distances[i] = np.inf
            nn[i] = np.argsort(distances)[:k]
        
        return nn
    
    def get_tour_length(self, tour: np.ndarray) -> float:
        """Calculate total tour length.
        
        Args:
            tour: Array of city indices
            
        Returns:
            Total tour length
        """
        return tour_length_from_matrix(tour, self.matrix)
    
    def get_tour_lengths_batch(self, tours: np.ndarray) -> np.ndarray:
        """Calculate tour lengths for multiple tours.
        
        Args:
            tours: Array of shape (n_tours, n_cities)
            
        Returns:
            Array of tour lengths
        """
        if self.use_gpu and self._gpu_matrix is not None:
            return self._tour_lengths_gpu(tours)
        return tour_lengths_batch(tours, self.matrix)
    
    def _tour_lengths_gpu(self, tours: np.ndarray) -> np.ndarray:
        """Calculate tour lengths on GPU."""
        import cupy as cp
        
        tours_gpu = cp.asarray(tours)
        n_tours = len(tours)
        lengths = cp.zeros(n_tours, dtype=cp.float64)
        
        for i in range(n_tours):
            tour = tours_gpu[i]
            length = 0.0
            for j in range(len(tour)):
                length += self._gpu_matrix[tour[j], tour[(j + 1) % len(tour)]]
            lengths[i] = length
        
        return cp.asnumpy(lengths)
    
    def get_edge_length(self, city1: int, city2: int) -> float:
        """Get distance between two cities.
        
        Args:
            city1: First city index
            city2: Second city index
            
        Returns:
            Distance between cities
        """
        return self.matrix[city1, city2]
    
    def get_longest_edge(self, tour: np.ndarray) -> Tuple[float, int, int]:
        """Find the longest edge in a tour.
        
        Args:
            tour: Array of city indices
            
        Returns:
            Tuple of (length, city1_idx, city2_idx)
        """
        return longest_edge_in_tour(tour, self.matrix)


@njit(cache=True, parallel=True)
def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix.
    
    Parallelized with Numba for large instances.
    
    Args:
        coords: City coordinates (n_cities, 2)
        
    Returns:
        Distance matrix (n_cities, n_cities)
    """
    n = len(coords)
    matrix = np.empty((n, n), dtype=np.float64)
    
    for i in prange(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                matrix[i, j] = np.sqrt(dx * dx + dy * dy)
    
    return matrix


@njit(cache=True)
def tour_length_from_matrix(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Calculate tour length using precomputed distance matrix.
    
    O(n) time complexity with O(1) distance lookups.
    
    Args:
        tour: Array of city indices
        dist_matrix: Precomputed distance matrix
        
    Returns:
        Total tour length
    """
    n = len(tour)
    total = 0.0
    
    for i in range(n):
        total += dist_matrix[tour[i], tour[(i + 1) % n]]
    
    return total


@njit(cache=True, parallel=True)
def tour_lengths_batch(tours: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """Calculate tour lengths for multiple tours in parallel.
    
    Args:
        tours: Array of shape (n_tours, n_cities)
        dist_matrix: Precomputed distance matrix
        
    Returns:
        Array of tour lengths
    """
    n_tours = len(tours)
    lengths = np.empty(n_tours, dtype=np.float64)
    
    for i in prange(n_tours):
        lengths[i] = tour_length_from_matrix(tours[i], dist_matrix)
    
    return lengths


@njit(cache=True)
def longest_edge_in_tour(tour: np.ndarray, dist_matrix: np.ndarray) -> Tuple[float, int, int]:
    """Find the longest edge in a tour.
    
    Args:
        tour: Array of city indices
        dist_matrix: Precomputed distance matrix
        
    Returns:
        Tuple of (max_length, from_idx, to_idx) in tour positions
    """
    n = len(tour)
    max_len = 0.0
    max_from = 0
    max_to = 1
    
    for i in range(n):
        j = (i + 1) % n
        length = dist_matrix[tour[i], tour[j]]
        if length > max_len:
            max_len = length
            max_from = i
            max_to = j
    
    return max_len, max_from, max_to


@njit(cache=True)
def delta_2opt(tour: np.ndarray, i: int, j: int, dist_matrix: np.ndarray) -> float:
    """Calculate the change in tour length from a 2-opt move.
    
    A 2-opt move reverses the segment tour[i+1:j+1].
    This function computes the delta without performing the reversal.
    
    Args:
        tour: Current tour
        i: First cut position
        j: Second cut position (j > i)
        dist_matrix: Precomputed distance matrix
        
    Returns:
        Change in tour length (negative = improvement)
    """
    n = len(tour)
    
    # Current edges
    a, b = tour[i], tour[(i + 1) % n]
    c, d = tour[j], tour[(j + 1) % n]
    
    # New edges after reversal
    old_dist = dist_matrix[a, b] + dist_matrix[c, d]
    new_dist = dist_matrix[a, c] + dist_matrix[b, d]
    
    return new_dist - old_dist


@njit(cache=True)
def apply_2opt_move(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply a 2-opt move (reverse segment).
    
    Args:
        tour: Current tour
        i: First cut position
        j: Second cut position
        
    Returns:
        New tour with segment reversed
    """
    new_tour = tour.copy()
    
    # Reverse segment from i+1 to j (inclusive)
    left = i + 1
    right = j
    while left < right:
        new_tour[left], new_tour[right] = new_tour[right], new_tour[left]
        left += 1
        right -= 1
    
    return new_tour


def create_distance_matrix(coords: np.ndarray, use_gpu: bool = False) -> DistanceMatrix:
    """Factory function to create a distance matrix.
    
    Args:
        coords: City coordinates
        use_gpu: Whether to attempt GPU acceleration
        
    Returns:
        DistanceMatrix instance
    """
    return DistanceMatrix(coords, use_gpu=use_gpu)
