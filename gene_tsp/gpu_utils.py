"""GPU-accelerated utilities using CuPy.

Provides optional GPU acceleration for distance matrix computation
and batch fitness evaluation. Falls back to NumPy/Numba if CuPy unavailable.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    if not CUPY_AVAILABLE:
        return False
    try:
        # Try to allocate a small array
        x = cp.zeros(10)
        del x
        return True
    except Exception:
        return False


class GPUDistanceMatrix:
    """GPU-accelerated distance matrix operations.
    
    Uses CuPy for GPU computation when available.
    Falls back to NumPy for CPU computation otherwise.
    """
    
    def __init__(self, coords: np.ndarray, force_cpu: bool = False):
        """Initialize GPU distance matrix.
        
        Args:
            coords: City coordinates (n_cities, 2)
            force_cpu: Force CPU computation even if GPU available
        """
        self.coords_cpu = np.asarray(coords, dtype=np.float64)
        self.n_cities = len(coords)
        self.use_gpu = is_gpu_available() and not force_cpu
        
        if self.use_gpu:
            self._init_gpu()
        else:
            self._init_cpu()
    
    def _init_gpu(self):
        """Initialize GPU resources."""
        self.coords_gpu = cp.asarray(self.coords_cpu)
        
        # Compute distance matrix on GPU
        diff = self.coords_gpu[:, np.newaxis, :] - self.coords_gpu[np.newaxis, :, :]
        self.matrix_gpu = cp.sqrt(cp.sum(diff ** 2, axis=2))
        
        # Keep CPU copy for compatibility
        self.matrix_cpu = cp.asnumpy(self.matrix_gpu)
    
    def _init_cpu(self):
        """Initialize CPU-only resources."""
        from gene_tsp.distance import compute_distance_matrix
        self.matrix_cpu = compute_distance_matrix(self.coords_cpu)
        self.matrix_gpu = None
        self.coords_gpu = None
    
    @property
    def matrix(self) -> np.ndarray:
        """Get distance matrix (CPU array)."""
        return self.matrix_cpu
    
    def tour_length(self, tour: np.ndarray) -> float:
        """Calculate tour length.
        
        Uses GPU if available, otherwise CPU.
        
        Args:
            tour: Array of city indices
            
        Returns:
            Total tour length
        """
        if self.use_gpu:
            return self._tour_length_gpu(tour)
        return self._tour_length_cpu(tour)
    
    def _tour_length_cpu(self, tour: np.ndarray) -> float:
        """Calculate tour length on CPU."""
        n = len(tour)
        total = 0.0
        for i in range(n):
            total += self.matrix_cpu[tour[i], tour[(i + 1) % n]]
        return total
    
    def _tour_length_gpu(self, tour: np.ndarray) -> float:
        """Calculate tour length on GPU."""
        tour_gpu = cp.asarray(tour)
        n = len(tour)
        
        # Get edge lengths
        from_cities = tour_gpu
        to_cities = cp.roll(tour_gpu, -1)
        lengths = self.matrix_gpu[from_cities, to_cities]
        
        return float(cp.sum(lengths))
    
    def batch_tour_lengths(self, tours: np.ndarray) -> np.ndarray:
        """Calculate tour lengths for multiple tours.
        
        Args:
            tours: Array of shape (n_tours, n_cities)
            
        Returns:
            Array of tour lengths
        """
        if self.use_gpu:
            return self._batch_lengths_gpu(tours)
        return self._batch_lengths_cpu(tours)
    
    def _batch_lengths_cpu(self, tours: np.ndarray) -> np.ndarray:
        """Calculate batch tour lengths on CPU."""
        from gene_tsp.distance import tour_lengths_batch
        return tour_lengths_batch(tours, self.matrix_cpu)
    
    def _batch_lengths_gpu(self, tours: np.ndarray) -> np.ndarray:
        """Calculate batch tour lengths on GPU.
        
        Vectorized GPU computation for maximum performance.
        """
        tours_gpu = cp.asarray(tours)
        n_tours, n_cities = tours_gpu.shape
        
        # Create indices for from and to cities
        from_cities = tours_gpu
        to_cities = cp.roll(tours_gpu, -1, axis=1)
        
        # Batch index into distance matrix
        # This is the key optimization - parallel memory access
        lengths = self.matrix_gpu[from_cities, to_cities]
        
        # Sum along tour dimension
        total_lengths = cp.sum(lengths, axis=1)
        
        return cp.asnumpy(total_lengths)
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for entire population.
        
        Convenience method that calls batch_tour_lengths.
        
        Args:
            population: Array of shape (pop_size, n_cities)
            
        Returns:
            Array of fitness values (tour lengths)
        """
        return self.batch_tour_lengths(population)
    
    def find_nearest_neighbors(self, k: int = 20) -> np.ndarray:
        """Find k nearest neighbors for each city.
        
        Uses GPU for sorting if available.
        
        Args:
            k: Number of neighbors to find
            
        Returns:
            Array of shape (n_cities, k) with neighbor indices
        """
        k = min(k, self.n_cities - 1)
        
        if self.use_gpu:
            # Set diagonal to infinity
            matrix = self.matrix_gpu.copy()
            cp.fill_diagonal(matrix, cp.inf)
            
            # Argsort each row
            sorted_indices = cp.argsort(matrix, axis=1)
            neighbors = sorted_indices[:, :k]
            
            return cp.asnumpy(neighbors)
        else:
            neighbors = np.empty((self.n_cities, k), dtype=np.int64)
            for i in range(self.n_cities):
                distances = self.matrix_cpu[i].copy()
                distances[i] = np.inf
                neighbors[i] = np.argsort(distances)[:k]
            return neighbors
    
    def to_cpu(self):
        """Ensure all data is on CPU (for compatibility)."""
        if self.matrix_gpu is not None:
            self.matrix_cpu = cp.asnumpy(self.matrix_gpu)
    
    def free_gpu_memory(self):
        """Free GPU memory."""
        if self.use_gpu:
            del self.matrix_gpu
            del self.coords_gpu
            self.matrix_gpu = None
            self.coords_gpu = None
            cp.get_default_memory_pool().free_all_blocks()


def gpu_two_opt_delta(tour: np.ndarray, dist_matrix_gpu, 
                      i: int, j: int) -> float:
    """Calculate 2-opt delta on GPU.
    
    Args:
        tour: Current tour (GPU array)
        dist_matrix_gpu: Distance matrix (GPU array)
        i, j: Cut positions
        
    Returns:
        Change in tour length (negative = improvement)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    n = len(tour)
    a, b = tour[i], tour[(i + 1) % n]
    c, d = tour[j], tour[(j + 1) % n]
    
    old_dist = dist_matrix_gpu[a, b] + dist_matrix_gpu[c, d]
    new_dist = dist_matrix_gpu[a, c] + dist_matrix_gpu[b, d]
    
    return float(new_dist - old_dist)


def create_distance_matrix(coords: np.ndarray, 
                           use_gpu: bool = True) -> 'GPUDistanceMatrix':
    """Factory function to create distance matrix.
    
    Args:
        coords: City coordinates
        use_gpu: Whether to attempt GPU acceleration
        
    Returns:
        GPUDistanceMatrix instance
    """
    return GPUDistanceMatrix(coords, force_cpu=not use_gpu)


# =============================================================================
# GPU-Accelerated Genetic Operators (for large populations)
# =============================================================================

def gpu_batch_mutation_inversion(population: np.ndarray, 
                                  mutation_rate: float) -> np.ndarray:
    """Apply inversion mutation to population on GPU.
    
    Each individual has mutation_rate probability of mutation.
    
    Args:
        population: Population array (pop_size, n_cities)
        mutation_rate: Probability of mutation
        
    Returns:
        Mutated population
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy not available, using CPU fallback")
        from gene_tsp.mutation import inversion_mutation
        result = population.copy()
        for i in range(len(result)):
            if np.random.random() < mutation_rate:
                result[i] = inversion_mutation(result[i])
        return result
    
    pop_gpu = cp.asarray(population)
    pop_size, n_cities = pop_gpu.shape
    
    # Generate random mutation decisions
    mutate_mask = cp.random.random(pop_size) < mutation_rate
    
    # Generate random cut points for each individual
    cuts = cp.random.randint(0, n_cities, size=(pop_size, 2))
    cuts = cp.sort(cuts, axis=1)
    
    # Apply mutations (this is still sequential per-individual for correctness)
    result = pop_gpu.copy()
    mutate_indices = cp.where(mutate_mask)[0]
    
    for idx in cp.asnumpy(mutate_indices):
        i, j = int(cuts[idx, 0]), int(cuts[idx, 1])
        # Reverse segment
        result[idx, i:j+1] = result[idx, i:j+1][::-1]
    
    return cp.asnumpy(result)


def benchmark_gpu_vs_cpu(n_cities: int = 1000, n_tours: int = 100):
    """Benchmark GPU vs CPU performance.
    
    Args:
        n_cities: Number of cities
        n_tours: Number of tours to evaluate
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Generate random data
    coords = np.random.rand(n_cities, 2)
    tours = np.array([np.random.permutation(n_cities) for _ in range(n_tours)])
    
    results = {}
    
    # CPU timing
    cpu_dm = GPUDistanceMatrix(coords, force_cpu=True)
    start = time.time()
    cpu_lengths = cpu_dm.batch_tour_lengths(tours)
    results['cpu_time'] = time.time() - start
    
    # GPU timing (if available)
    if is_gpu_available():
        gpu_dm = GPUDistanceMatrix(coords, force_cpu=False)
        
        # Warm-up
        _ = gpu_dm.batch_tour_lengths(tours[:10])
        
        start = time.time()
        gpu_lengths = gpu_dm.batch_tour_lengths(tours)
        results['gpu_time'] = time.time() - start
        
        # Verify correctness
        results['max_diff'] = np.max(np.abs(cpu_lengths - gpu_lengths))
        results['speedup'] = results['cpu_time'] / results['gpu_time']
        
        gpu_dm.free_gpu_memory()
    else:
        results['gpu_time'] = None
        results['speedup'] = None
        results['note'] = 'GPU not available'
    
    return results
