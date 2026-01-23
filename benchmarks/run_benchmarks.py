"""Benchmark suite for comparing TSP algorithms.

Runs multiple algorithms on various datasets and reports results.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gene_tsp.algorithms import (
    run_algorithm, list_presets, get_preset, AlgorithmConfig
)
from gene_tsp.tsplib import TSPLIBManager, OPTIMAL_SOLUTIONS
from data.generators import (
    generate_uniform, generate_clustered, generate_circle, generate_grid
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    algorithm: str
    dataset: str
    n_cities: int
    best_fitness: float
    optimal_fitness: Optional[float]
    gap_percent: Optional[float]
    runtime_seconds: float
    generations: int
    timestamp: str


def run_single_benchmark(coords: np.ndarray, algorithm: str,
                         optimal: Optional[float] = None,
                         dataset_name: str = "unknown") -> BenchmarkResult:
    """Run a single benchmark.
    
    Args:
        coords: City coordinates
        algorithm: Algorithm preset name
        optimal: Known optimal solution (if any)
        dataset_name: Name of the dataset
        
    Returns:
        BenchmarkResult
    """
    config = get_preset(algorithm)
    
    start_time = time.time()
    best_tour, best_fitness, stats = run_algorithm(
        coords, preset=algorithm, verbose=False
    )
    runtime = time.time() - start_time
    
    gap = None
    if optimal:
        gap = 100 * (best_fitness - optimal) / optimal
    
    return BenchmarkResult(
        algorithm=algorithm,
        dataset=dataset_name,
        n_cities=len(coords),
        best_fitness=best_fitness,
        optimal_fitness=optimal,
        gap_percent=gap,
        runtime_seconds=runtime,
        generations=config.n_generations,
        timestamp=datetime.now().isoformat()
    )


def run_algorithm_comparison(coords: np.ndarray,
                              algorithms: Optional[List[str]] = None,
                              optimal: Optional[float] = None,
                              dataset_name: str = "unknown",
                              n_runs: int = 1) -> List[BenchmarkResult]:
    """Compare multiple algorithms on the same dataset.
    
    Args:
        coords: City coordinates
        algorithms: List of algorithm preset names (None = all)
        optimal: Known optimal solution
        dataset_name: Dataset name
        n_runs: Number of runs per algorithm (for averaging)
        
    Returns:
        List of BenchmarkResult
    """
    if algorithms is None:
        algorithms = list(list_presets().keys())
    
    results = []
    
    for algo in algorithms:
        print(f"  Running {algo}...", end=" ", flush=True)
        
        best_result = None
        for run in range(n_runs):
            result = run_single_benchmark(coords, algo, optimal, dataset_name)
            if best_result is None or result.best_fitness < best_result.best_fitness:
                best_result = result
        
        results.append(best_result)
        gap_str = f"(gap: {best_result.gap_percent:.2f}%)" if best_result.gap_percent else ""
        print(f"{best_result.best_fitness:.2f} in {best_result.runtime_seconds:.1f}s {gap_str}")
    
    return results


def run_benchmark_suite(output_dir: str = "benchmarks/results"):
    """Run full benchmark suite.
    
    Tests algorithms on multiple datasets and saves results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define datasets
    datasets = [
        # Generated datasets
        ("uniform_50", generate_uniform(50, seed=42), None),
        ("uniform_100", generate_uniform(100, seed=42), None),
        ("uniform_200", generate_uniform(200, seed=42), None),
        ("clustered_50", generate_clustered(50, seed=42), None),
        ("clustered_100", generate_clustered(100, seed=42), None),
        ("circle_50", generate_circle(50), 2 * np.pi * 50 / 50),  # Approx optimal
    ]
    
    # Add TSPLIB instances
    tsplib = TSPLIBManager()
    for name in ['eil51', 'berlin52', 'att48']:
        try:
            instance = tsplib.get_instance(name, normalize=False)
            datasets.append((name, instance.coords, instance.optimal_length))
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
    
    # Algorithms to compare
    algorithms = ['fast', 'standard', 'quality', 'memetic', 'island']
    
    all_results = []
    
    print("=" * 60)
    print("TSP Algorithm Benchmark Suite")
    print("=" * 60)
    
    for dataset_name, coords, optimal in datasets:
        print(f"\nDataset: {dataset_name} ({len(coords)} cities)")
        if optimal:
            print(f"  Optimal: {optimal}")
        print("-" * 40)
        
        results = run_algorithm_comparison(
            coords, algorithms, optimal, dataset_name, n_runs=1
        )
        all_results.extend(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Group by algorithm
    by_algorithm: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.algorithm not in by_algorithm:
            by_algorithm[r.algorithm] = []
        by_algorithm[r.algorithm].append(r)
    
    print("\nAverage gap from optimal (lower is better):")
    print("-" * 40)
    
    for algo, algo_results in sorted(by_algorithm.items()):
        gaps = [r.gap_percent for r in algo_results if r.gap_percent is not None]
        if gaps:
            avg_gap = np.mean(gaps)
            print(f"  {algo:20s}: {avg_gap:6.2f}%")
    
    print("\nAverage runtime (seconds):")
    print("-" * 40)
    
    for algo, algo_results in sorted(by_algorithm.items()):
        times = [r.runtime_seconds for r in algo_results]
        avg_time = np.mean(times)
        print(f"  {algo:20s}: {avg_time:6.2f}s")


def compare_on_dataset(coords: np.ndarray, 
                        algorithms: Optional[List[str]] = None,
                        n_runs: int = 3,
                        verbose: bool = True) -> Dict:
    """Compare algorithms on a single dataset.
    
    Args:
        coords: City coordinates
        algorithms: Algorithm preset names (None = default set)
        n_runs: Number of runs for statistical significance
        verbose: Print results
        
    Returns:
        Dictionary with comparison results
    """
    if algorithms is None:
        algorithms = ['fast', 'standard', 'memetic', 'island']
    
    results = {}
    
    for algo in algorithms:
        fitness_values = []
        runtimes = []
        
        for _ in range(n_runs):
            start = time.time()
            _, best_fitness, _ = run_algorithm(coords, preset=algo, verbose=False)
            runtime = time.time() - start
            
            fitness_values.append(best_fitness)
            runtimes.append(runtime)
        
        results[algo] = {
            'best': min(fitness_values),
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'mean_runtime': np.mean(runtimes),
        }
        
        if verbose:
            print(f"{algo:15s}: best={results[algo]['best']:.2f}, "
                  f"mean={results[algo]['mean']:.2f} ± {results[algo]['std']:.2f}, "
                  f"time={results[algo]['mean_runtime']:.2f}s")
    
    return results


def scalability_benchmark(max_cities: int = 1000,
                           algorithm: str = 'standard') -> Dict:
    """Test algorithm scalability with increasing problem size.
    
    Args:
        max_cities: Maximum number of cities to test
        algorithm: Algorithm to test
        
    Returns:
        Dictionary with scalability results
    """
    sizes = [20, 50, 100, 200, 500]
    if max_cities >= 1000:
        sizes.append(1000)
    if max_cities >= 2000:
        sizes.append(2000)
    
    results = {'sizes': [], 'runtimes': [], 'fitness': []}
    
    print(f"Scalability benchmark for {algorithm}")
    print("-" * 40)
    
    for n in sizes:
        if n > max_cities:
            break
        
        coords = generate_uniform(n, seed=42)
        
        start = time.time()
        _, best_fitness, _ = run_algorithm(coords, preset=algorithm, verbose=False)
        runtime = time.time() - start
        
        results['sizes'].append(n)
        results['runtimes'].append(runtime)
        results['fitness'].append(best_fitness)
        
        print(f"  n={n:4d}: fitness={best_fitness:.2f}, time={runtime:.2f}s")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TSP Algorithm Benchmarks')
    parser.add_argument('--suite', action='store_true', help='Run full benchmark suite')
    parser.add_argument('--scalability', action='store_true', help='Run scalability test')
    parser.add_argument('--compare', action='store_true', help='Compare algorithms')
    parser.add_argument('--n-cities', type=int, default=100, help='Number of cities')
    parser.add_argument('--algorithm', type=str, default='standard', help='Algorithm preset')
    
    args = parser.parse_args()
    
    if args.suite:
        run_benchmark_suite()
    elif args.scalability:
        scalability_benchmark(max_cities=args.n_cities, algorithm=args.algorithm)
    elif args.compare:
        coords = generate_clustered(args.n_cities, seed=42)
        print(f"\nComparing algorithms on {args.n_cities} cities:")
        print("=" * 50)
        compare_on_dataset(coords, n_runs=3)
    else:
        print("Use --suite, --scalability, or --compare")
        print("Example: python run_benchmarks.py --suite")
