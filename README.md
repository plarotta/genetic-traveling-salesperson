# Genetic Traveling Salesperson

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/PyQt6-GUI-orange.svg" alt="PyQt6">
</p>

A high-performance, interactive genetic algorithm library for solving the **Traveling Salesperson Problem (TSP)**. Features state-of-the-art evolutionary algorithms, beautiful real-time visualization, and support for problems with 5000+ cities.

## Features

### Algorithms
- **Standard Genetic Algorithm** with configurable operators
- **Memetic Algorithm** (GA + local search hybridization)
- **Island Model** parallel GA with migration
- **NSGA-II** multi-objective optimization (distance + smoothness)
- **Adaptive parameter control** based on population diversity

### Operators
| Category | Operators |
|----------|-----------|
| **Crossover** | Order (OX), Partially Mapped (PMX), Edge Recombination (ERX) |
| **Mutation** | Swap, Insert, Inversion, Scramble, Displacement |
| **Local Search** | 2-opt, 3-opt, Or-opt, Lin-Kernighan (lite) |
| **Selection** | Tournament, Roulette Wheel, Rank-based, Elitism |

### Performance
- **Numba JIT** compilation for critical paths
- **Multiprocessing** for parallel fitness evaluation
- **GPU acceleration** via CuPy (optional)
- Precomputed distance matrices for O(1) lookups
- Handles 5000+ city instances

### Visualization
- Interactive PyQt6 GUI
- Real-time fitness evolution charts
- Population diversity heatmaps
- Pareto front visualization for multi-objective
- Click to add/drag/remove cities

## Installation

### Using Conda/Mamba (Recommended)
```bash
# Create environment
conda env create -f environment.yml
conda activate tsp-env

# Or with mamba (faster)
mamba env create -f environment.yml
```

### Using pip
```bash
pip install -r requirements.txt
```

### Optional: GPU Acceleration
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

## Quick Start

### Launch the GUI
```bash
python main.py
```

### Command Line
```bash
# Solve a 100-city TSP with memetic algorithm
python main.py --cli -n 100 -a memetic -v

# List available algorithm presets
python main.py --cli --list-presets

# Run on a custom dataset
python main.py --cli -d data/challenge.txt -a island
```

### Python API
```python
from gene_tsp import run_algorithm
import numpy as np

# Generate random cities
coords = np.random.rand(100, 2)

# Run optimization
best_tour, best_fitness, stats = run_algorithm(
    coords, 
    preset='memetic',  # or 'standard', 'island', 'nsga2'
    verbose=True
)

print(f"Best tour length: {best_fitness:.2f}")
```

## Algorithm Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| `fast` | Quick GA, small population | Rapid prototyping |
| `standard` | Balanced GA settings | General use |
| `quality` | Large population, ERX crossover | Best solutions |
| `memetic` | GA + 2-opt local search | High quality |
| `island` | 4-island parallel GA | Large problems |
| `nsga2_distance_edge` | Multi-objective optimization | Smooth routes |

## Project Structure

```
genetic-traveling-salesperson/
├── gene_tsp/
│   ├── __init__.py          # Package entry point
│   ├── algorithms.py        # Algorithm registry and presets
│   ├── crossover.py         # Crossover operators (OX, PMX, ERX)
│   ├── mutation.py          # Mutation operators
│   ├── selection.py         # Selection methods
│   ├── local_search.py      # 2-opt, 3-opt, Or-opt
│   ├── distance.py          # Distance matrix utilities
│   ├── island_model.py      # Parallel island GA
│   ├── nsga2.py             # Multi-objective optimizer
│   ├── gpu_utils.py         # CuPy acceleration
│   ├── tsplib.py            # TSPLIB file parser
│   └── gui/
│       ├── main_window.py   # Main application window
│       ├── canvas.py        # Interactive visualization
│       ├── controls.py      # Parameter controls
│       ├── charts.py        # Statistics dashboard
│       └── themes.py        # Dark/light themes
├── data/
│   ├── generators.py        # Dataset generators
│   ├── easy.txt            # 9 cities (circle)
│   ├── medium.txt          # 49 cities (circle)
│   ├── hard.txt            # 500 cities (circle)
│   └── challenge.txt       # 1000 cities (random)
├── benchmarks/
│   └── run_benchmarks.py    # Benchmark suite
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## Datasets

### Built-in Datasets
| Name | Cities | Type | Optimal |
|------|--------|------|---------|
| easy | 9 | Circle | 2π |
| medium | 49 | Circle | 2π |
| hard | 500 | Circle | 2π |
| challenge | 1000 | Random | ~11 |

### TSPLIB Support
Built-in support for classic TSPLIB benchmarks:
- `berlin52` (52 cities, optimal: 7542)
- `eil51` (51 cities, optimal: 426)
- `att48` (48 cities, optimal: 10628)

### Custom Datasets
```python
from data.generators import generate_custom

# Available distributions
coords = generate_custom(100, distribution='clustered', seed=42)
# Options: uniform, clustered, circle, grid, star, concentric, two_clusters
```

## Benchmarks

Run the benchmark suite:
```bash
# Full benchmark suite
python benchmarks/run_benchmarks.py --suite

# Compare algorithms on a single instance
python benchmarks/run_benchmarks.py --compare -n 200

# Scalability test
python benchmarks/run_benchmarks.py --scalability --max-cities 2000
```

## Multi-Objective Optimization

NSGA-II optimizes multiple objectives simultaneously:

```python
from gene_tsp.nsga2 import run_nsga2

results = run_nsga2(
    coords,
    objectives=['distance', 'longest_edge'],  # Pareto optimization
    n_generations=300
)

# Get Pareto front
pareto_solutions = results['pareto_solutions']
pareto_objectives = results['pareto_objectives']

# Get compromise solution (balanced trade-off)
compromise_tour = results['best_per_objective']['distance']['tour']
```

## GUI Features

The interactive GUI provides:

- **Canvas**: Click to add cities, drag to move, right-click to remove
- **Controls**: Adjust population, mutation rate, algorithm type
- **Statistics**: Real-time fitness charts, diversity tracking
- **Pareto View**: Interactive Pareto front for NSGA-II
- **Themes**: Dark and light mode support

## Contributing

Contributions welcome! Areas of interest:
- Additional crossover/mutation operators
- More local search heuristics
- Performance optimizations
- Additional visualization features

## References

- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning
- Deb, K. et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
- Lin, S. & Kernighan, B. W. (1973). An Effective Heuristic Algorithm for the TSP

## License

MIT License - see [LICENSE](LICENSE) for details.
