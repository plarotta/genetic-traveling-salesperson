#!/usr/bin/env python3
"""
TSP Genetic Algorithm Visualizer

A high-performance, interactive application for solving the Traveling 
Salesman Problem using various genetic algorithm approaches.

Features:
- Multiple crossover operators (OX, PMX, ERX)
- Local search integration (2-opt, 3-opt)
- Island model parallel GA
- NSGA-II multi-objective optimization
- Interactive city manipulation
- Real-time visualization
- GPU acceleration support

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # Command-line mode
    python main.py --benchmark  # Run benchmarks
"""

import sys
import argparse
import numpy as np


def run_gui():
    """Launch the graphical user interface."""
    try:
        from gene_tsp.gui.main_window import run_app
        run_app()
    except ImportError as e:
        print(f"Error: Could not import GUI components: {e}")
        print("Make sure PyQt6 is installed: pip install PyQt6")
        sys.exit(1)


def run_cli(args):
    """Run in command-line mode."""
    from gene_tsp.algorithms import run_algorithm, list_presets
    from data.generators import load_dataset, generate_custom
    
    # Load or generate dataset
    if args.dataset:
        coords = load_dataset(args.dataset)
        print(f"Loaded {len(coords)} cities from {args.dataset}")
    else:
        coords = generate_custom(args.n_cities, distribution=args.distribution, seed=args.seed)
        print(f"Generated {args.n_cities} cities ({args.distribution} distribution)")
    
    # Show available presets
    if args.list_presets:
        print("\nAvailable algorithm presets:")
        for name, desc in list_presets().items():
            print(f"  {name:20s} - {desc}")
        return
    
    # Run algorithm
    print(f"\nRunning {args.algorithm} algorithm...")
    print(f"  Population: {args.population_size}")
    print(f"  Generations: {args.generations}")
    print("-" * 40)
    
    best_tour, best_fitness, stats = run_algorithm(
        coords,
        preset=args.algorithm,
        verbose=args.verbose,
        population_size=args.population_size,
        n_generations=args.generations,
    )
    
    print("-" * 40)
    print(f"Best tour length: {best_fitness:.4f}")
    
    # Save results
    if args.output:
        np.savetxt(args.output, best_tour, fmt='%d')
        print(f"Tour saved to {args.output}")


def run_benchmark(args):
    """Run benchmark suite."""
    from benchmarks.run_benchmarks import (
        run_benchmark_suite, compare_on_dataset, scalability_benchmark
    )
    from data.generators import generate_clustered
    
    if args.scalability:
        scalability_benchmark(max_cities=args.max_cities, algorithm=args.algorithm)
    elif args.compare:
        coords = generate_clustered(args.n_cities, seed=args.seed)
        print(f"\nComparing algorithms on {args.n_cities} cities:")
        print("=" * 50)
        compare_on_dataset(coords, n_runs=3)
    else:
        run_benchmark_suite()


def main():
    parser = argparse.ArgumentParser(
        description="TSP Genetic Algorithm Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch GUI
  python main.py --cli -n 100 -a memetic  # Solve 100-city TSP with memetic GA
  python main.py --benchmark --compare    # Compare algorithms
  python main.py --cli --list-presets     # Show available algorithms
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--cli', action='store_true',
                           help='Run in command-line mode')
    mode_group.add_argument('--benchmark', action='store_true',
                           help='Run benchmarks')
    
    # Dataset options
    parser.add_argument('--dataset', '-d', type=str,
                       help='Path to dataset file')
    parser.add_argument('--n-cities', '-n', type=int, default=50,
                       help='Number of cities for generated dataset')
    parser.add_argument('--distribution', type=str, default='clustered',
                       choices=['uniform', 'clustered', 'circle', 'grid', 'star'],
                       help='City distribution for generated dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Algorithm options
    parser.add_argument('--algorithm', '-a', type=str, default='standard',
                       help='Algorithm preset to use')
    parser.add_argument('--population-size', '-p', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', '-g', type=int, default=500,
                       help='Number of generations')
    
    # CLI options
    parser.add_argument('--list-presets', action='store_true',
                       help='List available algorithm presets')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for best tour')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # Benchmark options
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability benchmark')
    parser.add_argument('--compare', action='store_true',
                       help='Compare algorithms')
    parser.add_argument('--max-cities', type=int, default=1000,
                       help='Maximum cities for scalability test')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args)
    elif args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == '__main__':
    main()
