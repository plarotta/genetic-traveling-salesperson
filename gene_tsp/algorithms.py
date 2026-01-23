"""Algorithm registry and presets for TSP optimization.

Provides easy access to different algorithm configurations and presets.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from gene_tsp.distance import DistanceMatrix
from gene_tsp.crossover import get_crossover_operator
from gene_tsp.mutation import get_mutation_operator, compound_mutation
from gene_tsp.selection import (tournament_selection_batch, get_elite,
                                 survival_selection, calculate_diversity)
from gene_tsp.local_search import two_opt, three_opt


class AlgorithmType(Enum):
    """Available algorithm types."""
    STANDARD_GA = "standard_ga"
    MEMETIC_GA = "memetic_ga"
    ISLAND_MODEL = "island_model"
    NSGA2 = "nsga2"


@dataclass
class AlgorithmConfig:
    """Configuration for a genetic algorithm run."""
    name: str
    algorithm_type: AlgorithmType = AlgorithmType.STANDARD_GA
    
    # Population
    population_size: int = 100
    n_generations: int = 500
    
    # Operators
    crossover_op: str = 'ox'
    mutation_op: str = 'inversion'
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    
    # Selection
    tournament_size: int = 3
    elite_size: int = 2
    
    # Local search (for memetic)
    use_local_search: bool = False
    local_search_method: str = '2opt'
    local_search_prob: float = 0.1
    
    # Adaptive parameters
    adaptive_mutation: bool = False
    min_mutation_rate: float = 0.05
    max_mutation_rate: float = 0.5
    
    # Island model
    n_islands: int = 4
    migration_interval: int = 20
    migration_size: int = 2
    
    # Multi-objective (NSGA-II)
    objectives: list = field(default_factory=lambda: ['distance'])
    
    # Performance
    use_gpu: bool = False


# Predefined algorithm configurations
ALGORITHM_PRESETS: Dict[str, AlgorithmConfig] = {
    'standard': AlgorithmConfig(
        name='Standard GA',
        algorithm_type=AlgorithmType.STANDARD_GA,
        population_size=100,
        n_generations=500,
        crossover_op='ox',
        mutation_op='inversion',
    ),
    
    'fast': AlgorithmConfig(
        name='Fast GA',
        algorithm_type=AlgorithmType.STANDARD_GA,
        population_size=50,
        n_generations=200,
        crossover_op='ox',
        mutation_op='inversion',
        crossover_rate=0.95,
        mutation_rate=0.3,
    ),
    
    'quality': AlgorithmConfig(
        name='High Quality GA',
        algorithm_type=AlgorithmType.STANDARD_GA,
        population_size=200,
        n_generations=1000,
        crossover_op='erx',  # Edge recombination is best for TSP
        mutation_op='inversion',
        crossover_rate=0.85,
        mutation_rate=0.15,
        elite_size=5,
    ),
    
    'memetic': AlgorithmConfig(
        name='Memetic GA (GA + 2-opt)',
        algorithm_type=AlgorithmType.MEMETIC_GA,
        population_size=50,
        n_generations=300,
        crossover_op='ox',
        mutation_op='inversion',
        use_local_search=True,
        local_search_method='2opt',
        local_search_prob=0.2,
    ),
    
    'memetic_aggressive': AlgorithmConfig(
        name='Aggressive Memetic',
        algorithm_type=AlgorithmType.MEMETIC_GA,
        population_size=30,
        n_generations=200,
        crossover_op='erx',
        mutation_op='inversion',
        use_local_search=True,
        local_search_method='2opt',
        local_search_prob=0.5,
    ),
    
    'island': AlgorithmConfig(
        name='Island Model',
        algorithm_type=AlgorithmType.ISLAND_MODEL,
        population_size=50,
        n_generations=500,
        n_islands=4,
        migration_interval=20,
        migration_size=2,
    ),
    
    'island_large': AlgorithmConfig(
        name='Large Island Model',
        algorithm_type=AlgorithmType.ISLAND_MODEL,
        population_size=100,
        n_generations=1000,
        n_islands=8,
        migration_interval=30,
        migration_size=3,
    ),
    
    'nsga2_distance_edge': AlgorithmConfig(
        name='NSGA-II (Distance + Longest Edge)',
        algorithm_type=AlgorithmType.NSGA2,
        population_size=100,
        n_generations=300,
        objectives=['distance', 'longest_edge'],
    ),
    
    'nsga2_distance_variance': AlgorithmConfig(
        name='NSGA-II (Distance + Edge Variance)',
        algorithm_type=AlgorithmType.NSGA2,
        population_size=100,
        n_generations=300,
        objectives=['distance', 'edge_variance'],
    ),
    
    'adaptive': AlgorithmConfig(
        name='Adaptive GA',
        algorithm_type=AlgorithmType.STANDARD_GA,
        population_size=100,
        n_generations=500,
        adaptive_mutation=True,
        min_mutation_rate=0.05,
        max_mutation_rate=0.5,
    ),
}


def get_preset(name: str) -> AlgorithmConfig:
    """Get a preset algorithm configuration.
    
    Args:
        name: Preset name (e.g., 'standard', 'memetic', 'island')
        
    Returns:
        AlgorithmConfig for the preset
    """
    if name not in ALGORITHM_PRESETS:
        available = list(ALGORITHM_PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return ALGORITHM_PRESETS[name]


def list_presets() -> Dict[str, str]:
    """List available presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {name: config.name for name, config in ALGORITHM_PRESETS.items()}


class GeneticAlgorithm:
    """Generic genetic algorithm runner.
    
    Executes a GA based on the provided configuration.
    """
    
    def __init__(self, coords: np.ndarray, config: Optional[AlgorithmConfig] = None):
        """Initialize GA.
        
        Args:
            coords: City coordinates (n_cities, 2)
            config: Algorithm configuration (uses 'standard' preset if None)
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_cities = len(coords)
        self.config = config or get_preset('standard')
        
        # Create distance matrix
        self.dist_matrix = DistanceMatrix(coords, use_gpu=self.config.use_gpu)
        
        # Setup operators
        self._setup_operators()
        
        # State
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.best_tour: Optional[np.ndarray] = None
        self.best_fitness = float('inf')
        self.generation = 0
        
        # History
        self.best_history = []
        self.avg_history = []
        self.diversity_history = []
    
    def _setup_operators(self):
        """Set up crossover and mutation operators."""
        self.crossover_func = get_crossover_operator(self.config.crossover_op)
        
        if self.config.mutation_op == 'compound':
            self.mutation_func = None  # Use compound_mutation
        else:
            self.mutation_func = get_mutation_operator(self.config.mutation_op)
    
    def initialize(self):
        """Initialize population."""
        pop_size = self.config.population_size
        self.population = np.array([
            np.random.permutation(self.n_cities) for _ in range(pop_size)
        ], dtype=np.int64)
        
        self.fitness = self.dist_matrix.get_tour_lengths_batch(self.population)
        self._sort_population()
        self._update_best()
    
    def _sort_population(self):
        """Sort population by fitness."""
        indices = np.argsort(self.fitness)
        self.population = self.population[indices]
        self.fitness = self.fitness[indices]
    
    def _update_best(self):
        """Update best solution."""
        if self.fitness[0] < self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.best_tour = self.population[0].copy()
    
    def evolve_generation(self):
        """Evolve for one generation."""
        cfg = self.config
        pop_size = cfg.population_size
        
        # Adaptive mutation rate
        mutation_rate = cfg.mutation_rate
        if cfg.adaptive_mutation:
            diversity = calculate_diversity(self.population)
            # Low diversity -> high mutation
            mutation_rate = cfg.max_mutation_rate - diversity * (
                cfg.max_mutation_rate - cfg.min_mutation_rate)
            mutation_rate = np.clip(mutation_rate, cfg.min_mutation_rate, cfg.max_mutation_rate)
        
        # Selection
        parent_indices = tournament_selection_batch(
            self.fitness, pop_size, cfg.tournament_size)
        
        # Create offspring
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = self.population[parent_indices[i]]
            p2 = self.population[parent_indices[min(i+1, pop_size-1)]]
            
            # Crossover
            if np.random.random() < cfg.crossover_rate:
                c1, c2 = self.crossover_func(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            if self.mutation_func:
                if np.random.random() < mutation_rate:
                    c1 = self.mutation_func(c1)
                if np.random.random() < mutation_rate:
                    c2 = self.mutation_func(c2)
            else:
                c1 = compound_mutation(c1, mutation_rate)
                c2 = compound_mutation(c2, mutation_rate)
            
            # Local search (memetic)
            if cfg.use_local_search:
                if np.random.random() < cfg.local_search_prob:
                    c1, _ = two_opt(c1, self.dist_matrix.matrix, max_iterations=5)
                if np.random.random() < cfg.local_search_prob:
                    c2, _ = two_opt(c2, self.dist_matrix.matrix, max_iterations=5)
            
            offspring.extend([c1, c2])
        
        offspring = np.array(offspring[:pop_size])
        offspring_fitness = self.dist_matrix.get_tour_lengths_batch(offspring)
        
        # Survival selection
        self.population, self.fitness = survival_selection(
            self.population, self.fitness,
            offspring, offspring_fitness,
            pop_size, cfg.elite_size, method='elitist'
        )
        
        self._sort_population()
        self._update_best()
        
        # Record history
        self.generation += 1
        self.best_history.append(self.best_fitness)
        self.avg_history.append(np.mean(self.fitness))
        self.diversity_history.append(calculate_diversity(self.population))
    
    def run(self, callback: Optional[Callable] = None,
            verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Run full evolution.
        
        Args:
            callback: Optional callback(self) called each generation
            verbose: Print progress
            
        Returns:
            Tuple of (best_tour, best_fitness)
        """
        self.initialize()
        
        for gen in range(self.config.n_generations):
            self.evolve_generation()
            
            if callback:
                callback(self)
            
            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen {gen + 1}/{self.config.n_generations}: "
                      f"Best = {self.best_fitness:.4f}, "
                      f"Avg = {self.avg_history[-1]:.4f}, "
                      f"Diversity = {self.diversity_history[-1]:.4f}")
        
        return self.best_tour, self.best_fitness
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_tour': self.best_tour,
            'best_history': self.best_history,
            'avg_history': self.avg_history,
            'diversity_history': self.diversity_history,
        }


def run_algorithm(coords: np.ndarray, preset: str = 'standard',
                  verbose: bool = True, **kwargs) -> Tuple[np.ndarray, float, Dict]:
    """Convenience function to run an algorithm.
    
    Args:
        coords: City coordinates
        preset: Algorithm preset name
        verbose: Print progress
        **kwargs: Override config parameters
        
    Returns:
        Tuple of (best_tour, best_fitness, statistics)
    """
    config = get_preset(preset)
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if config.algorithm_type == AlgorithmType.ISLAND_MODEL:
        from gene_tsp.island_model import run_island_model
        return run_island_model(coords, n_generations=config.n_generations,
                               n_islands=config.n_islands, verbose=verbose)
    
    elif config.algorithm_type == AlgorithmType.NSGA2:
        from gene_tsp.nsga2 import run_nsga2
        results = run_nsga2(coords, objectives=config.objectives,
                           n_generations=config.n_generations,
                           population_size=config.population_size,
                           verbose=verbose)
        # Return best distance solution
        best_idx = np.argmin(results['pareto_objectives'][:, 0])
        best_tour = results['pareto_solutions'][best_idx]
        best_fitness = results['pareto_objectives'][best_idx, 0]
        return best_tour, best_fitness, results
    
    else:
        ga = GeneticAlgorithm(coords, config)
        best_tour, best_fitness = ga.run(verbose=verbose)
        return best_tour, best_fitness, ga.get_statistics()
