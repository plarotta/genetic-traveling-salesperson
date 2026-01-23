"""Island Model Genetic Algorithm for TSP.

Runs multiple populations (islands) in parallel with periodic migration.
Each island can use different operator configurations for diversity.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import time

from gene_tsp.distance import DistanceMatrix, tour_length_from_matrix
from gene_tsp.crossover import order_crossover, pmx_crossover, edge_recombination_crossover
from gene_tsp.mutation import inversion_mutation, swap_mutation, compound_mutation
from gene_tsp.selection import (tournament_selection_batch, get_elite, 
                                 survival_selection, calculate_diversity)
from gene_tsp.local_search import two_opt


@dataclass
class IslandConfig:
    """Configuration for a single island."""
    population_size: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 3
    n_elite: int = 2
    crossover_op: str = 'ox'  # 'ox', 'pmx', 'erx'
    mutation_op: str = 'inversion'  # 'inversion', 'swap', 'compound'
    use_local_search: bool = False
    local_search_prob: float = 0.1


@dataclass
class MigrationConfig:
    """Configuration for migration between islands."""
    migration_interval: int = 20  # Generations between migrations
    migration_size: int = 2  # Number of individuals to migrate
    migration_topology: str = 'ring'  # 'ring', 'random', 'fully_connected'


class Island:
    """A single island/population in the island model."""
    
    def __init__(self, island_id: int, config: IslandConfig, 
                 dist_matrix: np.ndarray, n_cities: int):
        """Initialize island.
        
        Args:
            island_id: Unique identifier for this island
            config: Island configuration
            dist_matrix: Precomputed distance matrix
            n_cities: Number of cities in TSP instance
        """
        self.island_id = island_id
        self.config = config
        self.dist_matrix = dist_matrix
        self.n_cities = n_cities
        
        # Initialize population randomly
        self.population = np.array([
            np.random.permutation(n_cities) 
            for _ in range(config.population_size)
        ], dtype=np.int64)
        
        # Compute initial fitness
        self.fitness = np.array([
            tour_length_from_matrix(tour, dist_matrix)
            for tour in self.population
        ])
        
        # Sort by fitness
        indices = np.argsort(self.fitness)
        self.population = self.population[indices]
        self.fitness = self.fitness[indices]
        
        # Track statistics
        self.best_fitness_history = [self.fitness[0]]
        self.diversity_history = [calculate_diversity(self.population)]
        self.generation = 0
        
        # Set up operators
        self._setup_operators()
    
    def _setup_operators(self):
        """Set up crossover and mutation operators based on config."""
        crossover_ops = {
            'ox': order_crossover,
            'pmx': pmx_crossover,
            'erx': edge_recombination_crossover,
        }
        self.crossover_func = crossover_ops.get(
            self.config.crossover_op, order_crossover)
        
        mutation_ops = {
            'inversion': inversion_mutation,
            'swap': swap_mutation,
            'compound': compound_mutation,
        }
        self.mutation_func = mutation_ops.get(
            self.config.mutation_op, inversion_mutation)
    
    def evolve_generation(self) -> float:
        """Evolve population for one generation.
        
        Returns:
            Best fitness in current generation
        """
        cfg = self.config
        pop_size = cfg.population_size
        
        # Selection
        parent_indices = tournament_selection_batch(
            self.fitness, pop_size, cfg.tournament_size)
        
        # Create offspring
        offspring = []
        offspring_fitness = []
        
        for i in range(0, pop_size, 2):
            p1_idx = parent_indices[i]
            p2_idx = parent_indices[min(i + 1, pop_size - 1)]
            
            parent1 = self.population[p1_idx]
            parent2 = self.population[p2_idx]
            
            # Crossover
            if np.random.random() < cfg.crossover_rate:
                child1, child2 = self.crossover_func(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if cfg.mutation_op == 'compound':
                child1 = compound_mutation(child1, cfg.mutation_rate)
                child2 = compound_mutation(child2, cfg.mutation_rate)
            else:
                if np.random.random() < cfg.mutation_rate:
                    child1 = self.mutation_func(child1)
                if np.random.random() < cfg.mutation_rate:
                    child2 = self.mutation_func(child2)
            
            # Optional local search
            if cfg.use_local_search:
                if np.random.random() < cfg.local_search_prob:
                    child1, _ = two_opt(child1, self.dist_matrix, max_iterations=5)
                if np.random.random() < cfg.local_search_prob:
                    child2, _ = two_opt(child2, self.dist_matrix, max_iterations=5)
            
            offspring.append(child1)
            offspring.append(child2)
            offspring_fitness.append(tour_length_from_matrix(child1, self.dist_matrix))
            offspring_fitness.append(tour_length_from_matrix(child2, self.dist_matrix))
        
        offspring = np.array(offspring[:pop_size])
        offspring_fitness = np.array(offspring_fitness[:pop_size])
        
        # Survival selection with elitism
        self.population, self.fitness = survival_selection(
            self.population, self.fitness,
            offspring, offspring_fitness,
            pop_size, cfg.n_elite, method='elitist'
        )
        
        # Sort by fitness
        indices = np.argsort(self.fitness)
        self.population = self.population[indices]
        self.fitness = self.fitness[indices]
        
        self.generation += 1
        self.best_fitness_history.append(self.fitness[0])
        self.diversity_history.append(calculate_diversity(self.population))
        
        return self.fitness[0]
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best individual and fitness."""
        return self.population[0].copy(), self.fitness[0]
    
    def get_migrants(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get best individuals for migration.
        
        Args:
            n: Number of migrants to select
            
        Returns:
            Tuple of (individuals, fitness)
        """
        n = min(n, len(self.population))
        return self.population[:n].copy(), self.fitness[:n].copy()
    
    def receive_migrants(self, migrants: np.ndarray, migrant_fitness: np.ndarray):
        """Receive migrants from other islands.
        
        Replaces worst individuals if migrants are better.
        
        Args:
            migrants: Array of immigrant tours
            migrant_fitness: Fitness of immigrants
        """
        for i, (migrant, fit) in enumerate(zip(migrants, migrant_fitness)):
            # Replace worst if better
            if fit < self.fitness[-1]:
                self.population[-1] = migrant
                self.fitness[-1] = fit
        
        # Re-sort
        indices = np.argsort(self.fitness)
        self.population = self.population[indices]
        self.fitness = self.fitness[indices]


def evolve_island_worker(args: Tuple) -> Dict:
    """Worker function for parallel island evolution.
    
    Args:
        args: Tuple of (island_state, n_generations, dist_matrix)
        
    Returns:
        Dictionary with evolved island state
    """
    island_data, n_generations, dist_matrix, config = args
    
    # Reconstruct island from serialized state
    island = Island.__new__(Island)
    island.island_id = island_data['island_id']
    island.config = config
    island.dist_matrix = dist_matrix
    island.n_cities = island_data['n_cities']
    island.population = island_data['population']
    island.fitness = island_data['fitness']
    island.generation = island_data['generation']
    island.best_fitness_history = island_data['best_fitness_history']
    island.diversity_history = island_data['diversity_history']
    island._setup_operators()
    
    # Evolve
    for _ in range(n_generations):
        island.evolve_generation()
    
    # Return serialized state
    return {
        'island_id': island.island_id,
        'n_cities': island.n_cities,
        'population': island.population,
        'fitness': island.fitness,
        'generation': island.generation,
        'best_fitness_history': island.best_fitness_history,
        'diversity_history': island.diversity_history,
    }


class IslandModelGA:
    """Island Model Genetic Algorithm.
    
    Maintains multiple populations that evolve independently with
    periodic migration of good solutions between islands.
    """
    
    def __init__(self, coords: np.ndarray, 
                 n_islands: int = 4,
                 island_configs: Optional[List[IslandConfig]] = None,
                 migration_config: Optional[MigrationConfig] = None,
                 use_multiprocessing: bool = True,
                 n_workers: Optional[int] = None):
        """Initialize island model.
        
        Args:
            coords: City coordinates (n_cities, 2)
            n_islands: Number of islands
            island_configs: Configuration for each island (or use defaults)
            migration_config: Migration settings
            use_multiprocessing: Whether to use parallel processing
            n_workers: Number of worker processes (None = cpu_count)
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_cities = len(coords)
        self.n_islands = n_islands
        
        # Create distance matrix
        self.dist_matrix_obj = DistanceMatrix(coords)
        self.dist_matrix = self.dist_matrix_obj.matrix
        
        # Set up configurations
        if island_configs is None:
            # Create diverse island configurations
            island_configs = self._create_diverse_configs(n_islands)
        self.island_configs = island_configs
        
        if migration_config is None:
            migration_config = MigrationConfig()
        self.migration_config = migration_config
        
        # Create islands
        self.islands: List[Island] = []
        for i in range(n_islands):
            config = island_configs[i] if i < len(island_configs) else island_configs[0]
            island = Island(i, config, self.dist_matrix, self.n_cities)
            self.islands.append(island)
        
        # Multiprocessing setup
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers if n_workers else min(cpu_count(), n_islands)
        
        # Track global best
        self._update_global_best()
        self.generation = 0
        
        # History for plotting
        self.global_best_history = [self.global_best_fitness]
        self.island_best_history = [[isl.fitness[0]] for isl in self.islands]
    
    def _create_diverse_configs(self, n_islands: int) -> List[IslandConfig]:
        """Create diverse configurations for islands.
        
        Different islands use different operator combinations
        to explore the search space more effectively.
        """
        configs = [
            # Standard GA with OX crossover
            IslandConfig(
                population_size=50, crossover_op='ox', 
                mutation_op='inversion', crossover_rate=0.9, mutation_rate=0.15
            ),
            # Edge-focused with ERX
            IslandConfig(
                population_size=50, crossover_op='erx',
                mutation_op='inversion', crossover_rate=0.85, mutation_rate=0.2
            ),
            # Higher mutation for exploration
            IslandConfig(
                population_size=50, crossover_op='pmx',
                mutation_op='compound', crossover_rate=0.8, mutation_rate=0.4
            ),
            # Memetic (GA + local search)
            IslandConfig(
                population_size=30, crossover_op='ox',
                mutation_op='inversion', use_local_search=True, local_search_prob=0.2,
                crossover_rate=0.9, mutation_rate=0.1
            ),
        ]
        
        # Repeat or truncate to match n_islands
        while len(configs) < n_islands:
            configs.append(configs[len(configs) % 4])
        return configs[:n_islands]
    
    def _update_global_best(self):
        """Update global best solution from all islands."""
        self.global_best_tour = None
        self.global_best_fitness = float('inf')
        
        for island in self.islands:
            tour, fit = island.get_best()
            if fit < self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_tour = tour
    
    def _perform_migration(self):
        """Migrate individuals between islands."""
        cfg = self.migration_config
        n_migrants = cfg.migration_size
        
        if cfg.migration_topology == 'ring':
            # Ring topology: island i sends to island i+1
            migrants_list = [isl.get_migrants(n_migrants) for isl in self.islands]
            for i, island in enumerate(self.islands):
                src = (i - 1) % self.n_islands
                island.receive_migrants(*migrants_list[src])
        
        elif cfg.migration_topology == 'random':
            # Random pairs
            indices = np.random.permutation(self.n_islands)
            for i in range(0, self.n_islands - 1, 2):
                isl1, isl2 = self.islands[indices[i]], self.islands[indices[i + 1]]
                migrants1 = isl1.get_migrants(n_migrants)
                migrants2 = isl2.get_migrants(n_migrants)
                isl1.receive_migrants(*migrants2)
                isl2.receive_migrants(*migrants1)
        
        elif cfg.migration_topology == 'fully_connected':
            # All islands share best individuals
            all_migrants = []
            all_fitness = []
            for island in self.islands:
                m, f = island.get_migrants(n_migrants)
                all_migrants.extend(m)
                all_fitness.extend(f)
            
            # Sort and take best
            all_migrants = np.array(all_migrants)
            all_fitness = np.array(all_fitness)
            best_idx = np.argsort(all_fitness)[:n_migrants * 2]
            best_migrants = all_migrants[best_idx]
            best_fitness = all_fitness[best_idx]
            
            # Send to all islands
            for island in self.islands:
                island.receive_migrants(best_migrants, best_fitness)
    
    def evolve(self, n_generations: int, callback: Optional[Callable] = None,
               verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Run evolution for specified number of generations.
        
        Args:
            n_generations: Number of generations to evolve
            callback: Optional callback function called each generation
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_tour, best_fitness)
        """
        migration_interval = self.migration_config.migration_interval
        
        for gen in range(n_generations):
            self.generation += 1
            
            if self.use_multiprocessing and self.n_islands > 1:
                # Parallel evolution
                self._evolve_parallel(1)
            else:
                # Sequential evolution
                for island in self.islands:
                    island.evolve_generation()
            
            # Migration
            if self.generation % migration_interval == 0:
                self._perform_migration()
            
            # Update global best
            self._update_global_best()
            self.global_best_history.append(self.global_best_fitness)
            for i, island in enumerate(self.islands):
                self.island_best_history[i].append(island.fitness[0])
            
            # Callback
            if callback:
                callback(self)
            
            # Progress
            if verbose and (gen + 1) % 10 == 0:
                avg_diversity = np.mean([isl.diversity_history[-1] for isl in self.islands])
                print(f"Gen {self.generation}: Best = {self.global_best_fitness:.4f}, "
                      f"Avg Diversity = {avg_diversity:.4f}")
        
        return self.global_best_tour, self.global_best_fitness
    
    def _evolve_parallel(self, n_generations: int):
        """Evolve islands in parallel using multiprocessing."""
        # Serialize island states
        island_data = [{
            'island_id': isl.island_id,
            'n_cities': isl.n_cities,
            'population': isl.population,
            'fitness': isl.fitness,
            'generation': isl.generation,
            'best_fitness_history': isl.best_fitness_history,
            'diversity_history': isl.diversity_history,
        } for isl in self.islands]
        
        # Prepare arguments
        args = [(data, n_generations, self.dist_matrix, self.island_configs[i])
                for i, data in enumerate(island_data)]
        
        # Run in parallel
        with Pool(self.n_workers) as pool:
            results = pool.map(evolve_island_worker, args)
        
        # Update islands from results
        for i, result in enumerate(results):
            self.islands[i].population = result['population']
            self.islands[i].fitness = result['fitness']
            self.islands[i].generation = result['generation']
            self.islands[i].best_fitness_history = result['best_fitness_history']
            self.islands[i].diversity_history = result['diversity_history']
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        return {
            'generation': self.generation,
            'global_best_fitness': self.global_best_fitness,
            'island_best': [isl.fitness[0] for isl in self.islands],
            'island_diversity': [isl.diversity_history[-1] for isl in self.islands],
            'global_best_history': self.global_best_history,
        }


def run_island_model(coords: np.ndarray, n_generations: int = 500,
                     n_islands: int = 4, verbose: bool = True,
                     **kwargs) -> Tuple[np.ndarray, float, Dict]:
    """Convenience function to run island model GA.
    
    Args:
        coords: City coordinates
        n_generations: Number of generations
        n_islands: Number of islands
        verbose: Print progress
        **kwargs: Additional arguments for IslandModelGA
        
    Returns:
        Tuple of (best_tour, best_fitness, statistics)
    """
    model = IslandModelGA(coords, n_islands=n_islands, **kwargs)
    best_tour, best_fitness = model.evolve(n_generations, verbose=verbose)
    stats = model.get_statistics()
    
    return best_tour, best_fitness, stats
