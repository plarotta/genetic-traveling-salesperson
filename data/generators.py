"""Dataset generators for TSP instances.

Provides various city distribution patterns for testing and benchmarking.
"""

import numpy as np
from typing import Tuple, Optional, List
import os


def generate_uniform(n_cities: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate cities uniformly distributed in unit square.
    
    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_cities, 2) with city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n_cities, 2)


def generate_circle(n_cities: int, radius: float = 1.0,
                    noise: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate cities on a circle with optional noise.
    
    Args:
        n_cities: Number of cities
        radius: Circle radius
        noise: Standard deviation of noise to add
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    if noise > 0:
        x += np.random.normal(0, noise, n_cities)
        y += np.random.normal(0, noise, n_cities)
    
    return np.column_stack([x, y])


def generate_concentric_circles(n_cities: int, n_circles: int = 3,
                                 seed: Optional[int] = None) -> np.ndarray:
    """Generate cities on concentric circles.
    
    Args:
        n_cities: Total number of cities
        n_circles: Number of concentric circles
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    cities_per_circle = n_cities // n_circles
    coords = []
    
    for i in range(n_circles):
        radius = (i + 1) / n_circles
        n = cities_per_circle if i < n_circles - 1 else n_cities - len(coords)
        circle_coords = generate_circle(n, radius=radius)
        coords.append(circle_coords)
    
    return np.vstack(coords)


def generate_grid(n_cities: int, noise: float = 0.05,
                  seed: Optional[int] = None) -> np.ndarray:
    """Generate cities on a grid with optional noise.
    
    Args:
        n_cities: Number of cities (will be rounded to nearest square)
        noise: Amount of noise to add (as fraction of grid spacing)
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    side = int(np.ceil(np.sqrt(n_cities)))
    x = np.linspace(0, 1, side)
    y = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(x, y)
    
    coords = np.column_stack([xx.flatten(), yy.flatten()])[:n_cities]
    
    if noise > 0:
        spacing = 1.0 / (side - 1) if side > 1 else 1.0
        coords += np.random.normal(0, noise * spacing, coords.shape)
    
    return coords


def generate_clustered(n_cities: int, n_clusters: int = 5,
                        cluster_std: float = 0.05,
                        seed: Optional[int] = None) -> np.ndarray:
    """Generate cities in clusters (simulates real-world delivery routes).
    
    Args:
        n_cities: Total number of cities
        n_clusters: Number of clusters
        cluster_std: Standard deviation of cluster spread
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centers
    centers = np.random.rand(n_clusters, 2)
    
    # Assign cities to clusters
    cities_per_cluster = n_cities // n_clusters
    coords = []
    
    for i, center in enumerate(centers):
        n = cities_per_cluster if i < n_clusters - 1 else n_cities - len(coords)
        cluster_coords = center + np.random.normal(0, cluster_std, (n, 2))
        coords.extend(cluster_coords)
    
    return np.array(coords)


def generate_two_clusters(n_cities: int, separation: float = 0.5,
                          seed: Optional[int] = None) -> np.ndarray:
    """Generate cities in two clearly separated clusters.
    
    Useful for testing algorithm behavior on bi-modal distributions.
    
    Args:
        n_cities: Total number of cities
        separation: Distance between cluster centers
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    n1 = n_cities // 2
    n2 = n_cities - n1
    
    cluster1 = np.random.normal(loc=[0.25, 0.5], scale=0.1, size=(n1, 2))
    cluster2 = np.random.normal(loc=[0.25 + separation, 0.5], scale=0.1, size=(n2, 2))
    
    return np.vstack([cluster1, cluster2])


def generate_star(n_cities: int, n_arms: int = 5,
                  seed: Optional[int] = None) -> np.ndarray:
    """Generate cities in a star pattern.
    
    Args:
        n_cities: Number of cities
        n_arms: Number of star arms
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    coords = []
    cities_per_arm = n_cities // n_arms
    
    for i in range(n_arms):
        angle = 2 * np.pi * i / n_arms
        n = cities_per_arm if i < n_arms - 1 else n_cities - len(coords)
        
        # Points along the arm
        t = np.random.rand(n) * 0.8 + 0.1  # Avoid center
        noise = np.random.normal(0, 0.02, (n, 2))
        
        arm_coords = np.column_stack([
            0.5 + t * 0.4 * np.cos(angle),
            0.5 + t * 0.4 * np.sin(angle)
        ]) + noise
        
        coords.extend(arm_coords)
    
    return np.array(coords)


def generate_diagonal_stripe(n_cities: int, width: float = 0.1,
                             seed: Optional[int] = None) -> np.ndarray:
    """Generate cities along a diagonal stripe.
    
    Args:
        n_cities: Number of cities
        width: Width of the stripe
        seed: Random seed
        
    Returns:
        Array of city coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.random.rand(n_cities)
    offset = np.random.uniform(-width/2, width/2, n_cities)
    
    x = t + offset * 0.5
    y = t - offset * 0.5
    
    return np.column_stack([x, y])


# =============================================================================
# Real-World Datasets
# =============================================================================

# Major US cities with approximate coordinates (normalized to [0, 1])
US_CITIES = {
    'New York': (0.89, 0.70),
    'Los Angeles': (0.08, 0.37),
    'Chicago': (0.65, 0.67),
    'Houston': (0.45, 0.22),
    'Phoenix': (0.18, 0.32),
    'Philadelphia': (0.86, 0.67),
    'San Antonio': (0.40, 0.21),
    'San Diego': (0.09, 0.31),
    'Dallas': (0.47, 0.29),
    'San Jose': (0.05, 0.43),
    'Austin': (0.43, 0.24),
    'Jacksonville': (0.79, 0.26),
    'Fort Worth': (0.46, 0.30),
    'Columbus': (0.72, 0.60),
    'Charlotte': (0.78, 0.42),
    'San Francisco': (0.04, 0.46),
    'Indianapolis': (0.68, 0.58),
    'Seattle': (0.06, 0.85),
    'Denver': (0.32, 0.55),
    'Washington DC': (0.83, 0.61),
    'Boston': (0.92, 0.76),
    'Nashville': (0.68, 0.44),
    'Detroit': (0.72, 0.68),
    'Portland': (0.06, 0.80),
    'Memphis': (0.61, 0.40),
    'Las Vegas': (0.14, 0.42),
    'Louisville': (0.69, 0.51),
    'Baltimore': (0.83, 0.63),
    'Milwaukee': (0.64, 0.69),
    'Albuquerque': (0.26, 0.40),
    'Tucson': (0.19, 0.29),
    'Fresno': (0.07, 0.43),
    'Sacramento': (0.06, 0.50),
    'Kansas City': (0.52, 0.51),
    'Atlanta': (0.74, 0.37),
    'Miami': (0.82, 0.10),
    'New Orleans': (0.60, 0.21),
    'Cleveland': (0.75, 0.65),
    'Minneapolis': (0.54, 0.74),
    'Tampa': (0.78, 0.18),
    'St. Louis': (0.60, 0.52),
    'Pittsburgh': (0.78, 0.62),
    'Cincinnati': (0.71, 0.55),
    'Orlando': (0.80, 0.20),
    'Salt Lake City': (0.20, 0.58),
    'Omaha': (0.47, 0.61),
    'El Paso': (0.27, 0.28),
    'Boise': (0.13, 0.70),
}


def generate_usa_cities(include_all: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Generate dataset of major US cities.
    
    Args:
        include_all: If True, include all cities. Otherwise select subset.
        
    Returns:
        Tuple of (coordinates array, list of city names)
    """
    cities = list(US_CITIES.keys())
    coords = np.array([US_CITIES[city] for city in cities])
    return coords, cities


def generate_custom(n_cities: int, distribution: str = 'uniform',
                    seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Generate dataset with specified distribution.
    
    Args:
        n_cities: Number of cities
        distribution: One of 'uniform', 'circle', 'grid', 'clustered', 
                     'concentric', 'star', 'diagonal', 'two_clusters'
        seed: Random seed
        **kwargs: Additional arguments for specific distribution
        
    Returns:
        Array of city coordinates
    """
    generators = {
        'uniform': generate_uniform,
        'circle': generate_circle,
        'grid': generate_grid,
        'clustered': generate_clustered,
        'concentric': generate_concentric_circles,
        'star': generate_star,
        'diagonal': generate_diagonal_stripe,
        'two_clusters': generate_two_clusters,
    }
    
    if distribution not in generators:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Available: {list(generators.keys())}")
    
    return generators[distribution](n_cities, seed=seed, **kwargs)


def save_dataset(coords: np.ndarray, filepath: str, 
                 delimiter: str = ' ', city_names: Optional[List[str]] = None):
    """Save dataset to file.
    
    Args:
        coords: City coordinates
        filepath: Output file path
        delimiter: Delimiter between coordinates
        city_names: Optional list of city names to include as comments
    """
    with open(filepath, 'w') as f:
        if city_names:
            f.write(f"# {len(coords)} cities\n")
            f.write(f"# Names: {', '.join(city_names[:10])}{'...' if len(city_names) > 10 else ''}\n")
        
        for coord in coords:
            f.write(f"{coord[0]:.6f}{delimiter}{coord[1]:.6f}\n")


def load_dataset(filepath: str, delimiter: Optional[str] = None) -> np.ndarray:
    """Load dataset from file.
    
    Automatically detects delimiter.
    
    Args:
        filepath: Input file path
        delimiter: Delimiter (auto-detect if None)
        
    Returns:
        Array of city coordinates
    """
    # Try to detect delimiter
    with open(filepath, 'r') as f:
        first_line = f.readline()
        while first_line.startswith('#'):
            first_line = f.readline()
        
        if delimiter is None:
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = None  # whitespace
    
    return np.loadtxt(filepath, delimiter=delimiter, comments='#')


# =============================================================================
# Dataset Manager
# =============================================================================

class DatasetManager:
    """Manages TSP datasets with easy access and generation."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize dataset manager.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self._cache = {}
    
    def get_dataset(self, name: str) -> np.ndarray:
        """Get a dataset by name.
        
        Args:
            name: Dataset name (e.g., 'easy', 'medium', 'hard', 'challenge')
                  or 'uniform_100', 'clustered_500', etc.
                  
        Returns:
            Array of city coordinates
        """
        if name in self._cache:
            return self._cache[name].copy()
        
        # Check for generated dataset specification
        if '_' in name:
            parts = name.split('_')
            dist = parts[0]
            n = int(parts[1])
            seed = int(parts[2]) if len(parts) > 2 else 42
            
            coords = generate_custom(n, distribution=dist, seed=seed)
            self._cache[name] = coords
            return coords.copy()
        
        # Check for file
        for ext in ['', '.txt', '.tsp']:
            filepath = os.path.join(self.data_dir, name + ext)
            if os.path.exists(filepath):
                coords = load_dataset(filepath)
                self._cache[name] = coords
                return coords.copy()
        
        raise ValueError(f"Dataset not found: {name}")
    
    def list_datasets(self) -> List[str]:
        """List available dataset files."""
        if not os.path.exists(self.data_dir):
            return []
        
        datasets = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt') or filename.endswith('.tsp'):
                datasets.append(filename.rsplit('.', 1)[0])
        return sorted(datasets)
    
    def generate_benchmark_suite(self, sizes: List[int] = [50, 100, 200, 500, 1000]):
        """Generate a suite of benchmark datasets.
        
        Args:
            sizes: List of city counts to generate
        """
        distributions = ['uniform', 'clustered', 'circle', 'grid']
        
        for size in sizes:
            for dist in distributions:
                name = f"{dist}_{size}"
                filepath = os.path.join(self.data_dir, f"{name}.txt")
                
                if not os.path.exists(filepath):
                    coords = generate_custom(size, distribution=dist, seed=42)
                    save_dataset(coords, filepath)
                    print(f"Generated {name}")
