"""NSGA-II Multi-Objective Genetic Algorithm for TSP.

Implements Non-dominated Sorting Genetic Algorithm II for optimizing
multiple objectives simultaneously (e.g., total distance and longest edge).
"""

import numpy as np
from numba import njit, prange
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass

from gene_tsp.distance import DistanceMatrix, tour_length_from_matrix
from gene_tsp.crossover import order_crossover
from gene_tsp.mutation import inversion_mutation
from gene_tsp.local_search import two_opt


@dataclass
class NSGA2Config:
    """Configuration for NSGA-II."""
    population_size: int = 100
    n_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 2
    use_local_search: bool = False
    local_search_prob: float = 0.05


# =============================================================================
# Objective Functions
# =============================================================================

@njit(cache=True)
def objective_total_distance(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Total tour distance (standard TSP objective)."""
    return tour_length_from_matrix(tour, dist_matrix)


@njit(cache=True)
def objective_longest_edge(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Longest edge in the tour (for smoother routes)."""
    n = len(tour)
    max_edge = 0.0
    for i in range(n):
        edge = dist_matrix[tour[i], tour[(i + 1) % n]]
        if edge > max_edge:
            max_edge = edge
    return max_edge


@njit(cache=True)
def objective_edge_variance(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Variance of edge lengths (for balanced routes)."""
    n = len(tour)
    edges = np.empty(n, dtype=np.float64)
    for i in range(n):
        edges[i] = dist_matrix[tour[i], tour[(i + 1) % n]]
    return np.var(edges)


@njit(cache=True)
def count_crossings(tour: np.ndarray, coords: np.ndarray) -> int:
    """Count number of edge crossings (self-intersections).
    
    Uses line segment intersection test.
    """
    n = len(tour)
    crossings = 0
    
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            
            # Edge i: tour[i] -> tour[i+1]
            # Edge j: tour[j] -> tour[(j+1) % n]
            p1 = coords[tour[i]]
            p2 = coords[tour[(i + 1) % n]]
            p3 = coords[tour[j]]
            p4 = coords[tour[(j + 1) % n]]
            
            # Check intersection using cross products
            d1 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
            d2 = (p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])
            d3 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            d4 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                crossings += 1
    
    return crossings


# =============================================================================
# NSGA-II Core Algorithms
# =============================================================================

@njit(cache=True)
def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Check if obj1 dominates obj2 (all objectives, minimization).
    
    obj1 dominates obj2 if obj1 is no worse in all objectives
    and strictly better in at least one.
    """
    dominated = False
    at_least_one_better = False
    
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            dominated = True
            break
        if obj1[i] < obj2[i]:
            at_least_one_better = True
    
    return not dominated and at_least_one_better


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sorting.
    
    Sorts population into Pareto fronts.
    
    Args:
        objectives: Array of shape (pop_size, n_objectives)
        
    Returns:
        List of fronts, each front is a list of individual indices
    """
    n = len(objectives)
    
    # Domination count and dominated set for each individual
    domination_count = np.zeros(n, dtype=np.int64)
    dominated_by = [[] for _ in range(n)]  # Who dominates i
    dominates_list = [[] for _ in range(n)]  # Who i dominates
    
    fronts = [[]]
    
    # Calculate domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominates_list[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominates_list[j].append(i)
                domination_count[i] += 1
    
    # Find first front (non-dominated individuals)
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Generate subsequent fronts
    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        front_idx += 1
    
    # Remove empty last front
    if len(fronts[-1]) == 0:
        fronts.pop()
    
    return fronts


@njit(cache=True)
def crowding_distance(objectives: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Calculate crowding distance for individuals in a front.
    
    Individuals at the boundary get infinite distance.
    Others get distance based on objective space spread.
    
    Args:
        objectives: Full objectives array (pop_size, n_objectives)
        front: Array of indices in this front
        
    Returns:
        Array of crowding distances for front members
    """
    n_front = len(front)
    n_obj = objectives.shape[1]
    
    if n_front <= 2:
        return np.full(n_front, np.inf)
    
    distances = np.zeros(n_front)
    
    for obj_idx in range(n_obj):
        # Sort front by this objective
        obj_values = objectives[front, obj_idx]
        sorted_indices = np.argsort(obj_values)
        
        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Calculate spread
        obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if obj_range == 0:
            continue
        
        # Interior points
        for i in range(1, n_front - 1):
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]
            distances[sorted_indices[i]] += (obj_values[next_idx] - obj_values[prev_idx]) / obj_range
    
    return distances


@njit(cache=True)
def crowded_comparison(rank1: int, dist1: float, rank2: int, dist2: float) -> int:
    """Crowded comparison operator.
    
    Prefers lower rank. If same rank, prefers larger crowding distance.
    
    Returns:
        -1 if individual 1 is better, 1 if individual 2 is better, 0 if equal
    """
    if rank1 < rank2:
        return -1
    elif rank1 > rank2:
        return 1
    elif dist1 > dist2:  # Same rank, prefer more crowded (larger distance)
        return -1
    elif dist1 < dist2:
        return 1
    return 0


@njit(cache=True)
def binary_tournament_nsga2(ranks: np.ndarray, distances: np.ndarray) -> int:
    """Binary tournament selection for NSGA-II.
    
    Selects based on rank first, then crowding distance.
    
    Args:
        ranks: Pareto rank for each individual
        distances: Crowding distance for each individual
        
    Returns:
        Index of selected individual
    """
    n = len(ranks)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    
    result = crowded_comparison(ranks[i], distances[i], ranks[j], distances[j])
    
    if result <= 0:
        return i
    return j


# =============================================================================
# NSGA-II Algorithm
# =============================================================================

class NSGA2:
    """NSGA-II Multi-Objective Optimizer for TSP.
    
    Optimizes multiple objectives simultaneously and maintains
    a diverse set of Pareto-optimal solutions.
    """
    
    def __init__(self, coords: np.ndarray, 
                 objectives: List[str] = ['distance', 'longest_edge'],
                 config: Optional[NSGA2Config] = None):
        """Initialize NSGA-II.
        
        Args:
            coords: City coordinates (n_cities, 2)
            objectives: List of objective names to optimize
            config: Algorithm configuration
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_cities = len(coords)
        
        # Create distance matrix
        self.dist_matrix_obj = DistanceMatrix(coords)
        self.dist_matrix = self.dist_matrix_obj.matrix
        
        # Configuration
        self.config = config if config else NSGA2Config()
        
        # Set up objectives
        self.objective_names = objectives
        self.n_objectives = len(objectives)
        self._setup_objectives()
        
        # Initialize population
        self._initialize_population()
        
        # History for visualization
        self.pareto_front_history: List[np.ndarray] = []
        self.hypervolume_history: List[float] = []
    
    def _setup_objectives(self):
        """Set up objective functions."""
        self.objective_funcs = []
        
        for name in self.objective_names:
            if name == 'distance':
                self.objective_funcs.append(
                    lambda t: objective_total_distance(t, self.dist_matrix))
            elif name == 'longest_edge':
                self.objective_funcs.append(
                    lambda t: objective_longest_edge(t, self.dist_matrix))
            elif name == 'edge_variance':
                self.objective_funcs.append(
                    lambda t: objective_edge_variance(t, self.dist_matrix))
            elif name == 'crossings':
                self.objective_funcs.append(
                    lambda t: count_crossings(t, self.coords))
            else:
                raise ValueError(f"Unknown objective: {name}")
    
    def _initialize_population(self):
        """Initialize random population and evaluate objectives."""
        pop_size = self.config.population_size
        
        self.population = np.array([
            np.random.permutation(self.n_cities)
            for _ in range(pop_size)
        ], dtype=np.int64)
        
        self.objectives = self._evaluate_objectives(self.population)
        
        # Initial sorting
        self._assign_ranks_and_distances()
    
    def _evaluate_objectives(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all objectives for population."""
        n = len(population)
        objectives = np.zeros((n, self.n_objectives))
        
        for i in range(n):
            for j, func in enumerate(self.objective_funcs):
                objectives[i, j] = func(population[i])
        
        return objectives
    
    def _assign_ranks_and_distances(self):
        """Assign Pareto ranks and crowding distances."""
        fronts = fast_non_dominated_sort(self.objectives)
        
        n = len(self.population)
        self.ranks = np.zeros(n, dtype=np.int64)
        self.crowding_distances = np.zeros(n)
        
        for rank, front in enumerate(fronts):
            front_array = np.array(front, dtype=np.int64)
            for idx in front:
                self.ranks[idx] = rank
            
            if len(front) > 0:
                distances = crowding_distance(self.objectives, front_array)
                for i, idx in enumerate(front):
                    self.crowding_distances[idx] = distances[i]
    
    def _select_parents(self, n_parents: int) -> np.ndarray:
        """Select parents using binary tournament."""
        parents = np.empty(n_parents, dtype=np.int64)
        for i in range(n_parents):
            parents[i] = binary_tournament_nsga2(self.ranks, self.crowding_distances)
        return parents
    
    def _create_offspring(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create offspring through crossover and mutation."""
        cfg = self.config
        pop_size = cfg.population_size
        
        parent_indices = self._select_parents(pop_size)
        offspring = []
        
        for i in range(0, pop_size, 2):
            p1 = self.population[parent_indices[i]]
            p2 = self.population[parent_indices[min(i + 1, pop_size - 1)]]
            
            # Crossover
            if np.random.random() < cfg.crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            if np.random.random() < cfg.mutation_rate:
                c1 = inversion_mutation(c1)
            if np.random.random() < cfg.mutation_rate:
                c2 = inversion_mutation(c2)
            
            # Optional local search
            if cfg.use_local_search and np.random.random() < cfg.local_search_prob:
                c1, _ = two_opt(c1, self.dist_matrix, max_iterations=3)
            if cfg.use_local_search and np.random.random() < cfg.local_search_prob:
                c2, _ = two_opt(c2, self.dist_matrix, max_iterations=3)
            
            offspring.append(c1)
            offspring.append(c2)
        
        offspring = np.array(offspring[:pop_size])
        offspring_objectives = self._evaluate_objectives(offspring)
        
        return offspring, offspring_objectives
    
    def _survivor_selection(self, combined_pop: np.ndarray, 
                           combined_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select survivors for next generation."""
        pop_size = self.config.population_size
        
        # Sort into fronts
        fronts = fast_non_dominated_sort(combined_obj)
        
        new_pop = []
        new_obj = []
        
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                # Add entire front
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
            else:
                # Need to select subset based on crowding distance
                remaining = pop_size - len(new_pop)
                if remaining > 0:
                    front_array = np.array(front, dtype=np.int64)
                    distances = crowding_distance(combined_obj, front_array)
                    
                    # Sort by crowding distance (descending) and take top
                    sorted_indices = np.argsort(-distances)
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                break
        
        return np.array(new_pop), np.array(new_obj)
    
    def evolve(self, n_generations: Optional[int] = None,
               callback: Optional[Callable] = None,
               verbose: bool = True) -> Dict:
        """Run NSGA-II evolution.
        
        Args:
            n_generations: Number of generations (uses config if None)
            callback: Optional callback(self) each generation
            verbose: Print progress
            
        Returns:
            Dictionary with Pareto front and statistics
        """
        if n_generations is None:
            n_generations = self.config.n_generations
        
        for gen in range(n_generations):
            # Create offspring
            offspring, offspring_obj = self._create_offspring()
            
            # Combine parent and offspring populations
            combined_pop = np.vstack([self.population, offspring])
            combined_obj = np.vstack([self.objectives, offspring_obj])
            
            # Survivor selection
            self.population, self.objectives = self._survivor_selection(
                combined_pop, combined_obj)
            
            # Update ranks and distances
            self._assign_ranks_and_distances()
            
            # Store Pareto front
            pareto_front_indices = np.where(self.ranks == 0)[0]
            self.pareto_front_history.append(self.objectives[pareto_front_indices].copy())
            
            # Callback
            if callback:
                callback(self)
            
            # Progress
            if verbose and (gen + 1) % 20 == 0:
                n_pareto = np.sum(self.ranks == 0)
                best_dist = np.min(self.objectives[:, 0])
                print(f"Gen {gen + 1}: Pareto front size = {n_pareto}, "
                      f"Best distance = {best_dist:.4f}")
        
        return self.get_results()
    
    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current Pareto front.
        
        Returns:
            Tuple of (solutions, objectives) for Pareto-optimal individuals
        """
        pareto_indices = np.where(self.ranks == 0)[0]
        return self.population[pareto_indices].copy(), self.objectives[pareto_indices].copy()
    
    def get_results(self) -> Dict:
        """Get final results.
        
        Returns:
            Dictionary with Pareto front, best solutions for each objective, etc.
        """
        pareto_solutions, pareto_objectives = self.get_pareto_front()
        
        # Find best for each objective
        best_per_objective = {}
        for i, name in enumerate(self.objective_names):
            best_idx = np.argmin(pareto_objectives[:, i])
            best_per_objective[name] = {
                'tour': pareto_solutions[best_idx],
                'objectives': pareto_objectives[best_idx],
            }
        
        return {
            'pareto_solutions': pareto_solutions,
            'pareto_objectives': pareto_objectives,
            'objective_names': self.objective_names,
            'best_per_objective': best_per_objective,
            'pareto_front_history': self.pareto_front_history,
        }
    
    def get_compromise_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get compromise solution (closest to ideal point).
        
        Returns:
            Tuple of (tour, objectives) for the compromise solution
        """
        pareto_solutions, pareto_objectives = self.get_pareto_front()
        
        # Normalize objectives
        min_vals = np.min(pareto_objectives, axis=0)
        max_vals = np.max(pareto_objectives, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        normalized = (pareto_objectives - min_vals) / range_vals
        
        # Find point closest to origin (ideal point)
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        best_idx = np.argmin(distances)
        
        return pareto_solutions[best_idx], pareto_objectives[best_idx]


def run_nsga2(coords: np.ndarray, 
              objectives: List[str] = ['distance', 'longest_edge'],
              n_generations: int = 200,
              population_size: int = 100,
              verbose: bool = True) -> Dict:
    """Convenience function to run NSGA-II.
    
    Args:
        coords: City coordinates
        objectives: List of objective names
        n_generations: Number of generations
        population_size: Population size
        verbose: Print progress
        
    Returns:
        Results dictionary with Pareto front
    """
    config = NSGA2Config(
        population_size=population_size,
        n_generations=n_generations
    )
    
    nsga2 = NSGA2(coords, objectives=objectives, config=config)
    results = nsga2.evolve(verbose=verbose)
    
    return results
