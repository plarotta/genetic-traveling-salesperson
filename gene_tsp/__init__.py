"""
Gene TSP - Genetic Algorithm Library for the Traveling Salesman Problem

A comprehensive library implementing various evolutionary algorithms
for solving the Traveling Salesman Problem, including:

- Standard Genetic Algorithm
- Memetic Algorithm (GA + Local Search)
- Island Model Parallel GA
- NSGA-II Multi-Objective Optimization

Features:
- Multiple crossover operators (OX, PMX, Edge Recombination)
- Multiple mutation operators (swap, insert, inversion, scramble)
- Local search (2-opt, 3-opt, Or-opt)
- Tournament selection with elitism
- Adaptive parameter control
- GPU acceleration support (optional)
- Interactive PyQt6 visualization

Example:
    >>> from gene_tsp.algorithms import run_algorithm
    >>> import numpy as np
    >>> coords = np.random.rand(50, 2)
    >>> best_tour, best_fitness, stats = run_algorithm(coords, preset='memetic')
    >>> print(f"Best tour length: {best_fitness:.2f}")
"""

__version__ = '2.0.0'
__author__ = 'Pedro Larotta'

# Core imports for convenience
from gene_tsp.distance import DistanceMatrix, create_distance_matrix
from gene_tsp.algorithms import (
    run_algorithm, 
    get_preset, 
    list_presets,
    GeneticAlgorithm,
    AlgorithmConfig,
)

__all__ = [
    'DistanceMatrix',
    'create_distance_matrix',
    'run_algorithm',
    'get_preset',
    'list_presets',
    'GeneticAlgorithm',
    'AlgorithmConfig',
]
