"""TSPLIB file format parser and benchmark instances.

Supports reading standard TSPLIB format files (.tsp).
Includes known optimal solutions for benchmarking.
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TSPInstance:
    """A TSP problem instance."""
    name: str
    comment: str
    dimension: int
    coords: np.ndarray
    edge_weight_type: str
    optimal_length: Optional[float] = None
    optimal_tour: Optional[np.ndarray] = None


# Known optimal tour lengths for common TSPLIB instances
OPTIMAL_SOLUTIONS = {
    'berlin52': 7542,
    'kroA100': 21282,
    'kroA150': 26524,
    'kroA200': 29368,
    'kroB100': 22141,
    'kroB150': 26130,
    'kroB200': 29437,
    'kroC100': 20749,
    'kroD100': 21294,
    'kroE100': 22068,
    'eil51': 426,
    'eil76': 538,
    'eil101': 629,
    'st70': 675,
    'pr76': 108159,
    'pr107': 44303,
    'pr124': 59030,
    'pr136': 96772,
    'pr144': 58537,
    'pr152': 73682,
    'pr226': 80369,
    'pr264': 49135,
    'pr299': 48191,
    'pr439': 107217,
    'pr1002': 259045,
    'rat99': 1211,
    'rat195': 2323,
    'rat575': 6773,
    'rat783': 8806,
    'd198': 15780,
    'd493': 35002,
    'd657': 48912,
    'd1291': 50801,
    'd1655': 62128,
    'd2103': 80450,
    'lin105': 14379,
    'lin318': 42029,
    'ch130': 6110,
    'ch150': 6528,
    'a280': 2579,
    'ali535': 202339,
    'att48': 10628,
    'att532': 27686,
    'fl417': 11861,
    'fl1400': 20127,
    'fl1577': 22249,
    'fl3795': 28772,
    'fnl4461': 182566,
    'gil262': 2378,
    'gr96': 55209,
    'gr120': 6942,
    'gr137': 69853,
    'gr202': 40160,
    'gr229': 134602,
    'gr431': 171414,
    'gr666': 294358,
    'nrw1379': 56638,
    'p654': 34643,
    'pcb442': 50778,
    'pcb1173': 56892,
    'pcb3038': 137694,
    'pla7397': 23260728,
    'pla33810': 66048945,
    'pla85900': 142382641,
    'rd100': 7910,
    'rd400': 15281,
    'rl1304': 252948,
    'rl1323': 270199,
    'rl1889': 316536,
    'rl5915': 565530,
    'rl5934': 556045,
    'rl11849': 923288,
    'ts225': 126643,
    'tsp225': 3916,
    'u159': 42080,
    'u574': 36905,
    'u724': 41910,
    'u1060': 224094,
    'u1432': 152970,
    'u1817': 57201,
    'u2152': 64253,
    'u2319': 234256,
    'usa13509': 19982859,
    'vm1084': 239297,
    'vm1748': 336556,
}


def parse_tsplib(filepath: str) -> TSPInstance:
    """Parse a TSPLIB format file.
    
    Args:
        filepath: Path to .tsp file
        
    Returns:
        TSPInstance with parsed data
    """
    name = ""
    comment = ""
    dimension = 0
    edge_weight_type = "EUC_2D"
    coords = []
    node_coord_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('NAME'):
                name = line.split(':')[-1].strip()
            elif line.startswith('COMMENT'):
                comment = line.split(':')[-1].strip()
            elif line.startswith('DIMENSION'):
                dimension = int(line.split(':')[-1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(':')[-1].strip()
            elif line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
            elif line == 'EOF':
                break
            elif node_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: node_id x y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])
    
    coords = np.array(coords, dtype=np.float64)
    
    # Get optimal solution if known
    basename = os.path.splitext(os.path.basename(filepath))[0].lower()
    optimal = OPTIMAL_SOLUTIONS.get(basename)
    
    return TSPInstance(
        name=name,
        comment=comment,
        dimension=dimension,
        coords=coords,
        edge_weight_type=edge_weight_type,
        optimal_length=optimal
    )


def parse_tour_file(filepath: str) -> np.ndarray:
    """Parse a TSPLIB tour file (.opt.tour).
    
    Args:
        filepath: Path to tour file
        
    Returns:
        Array of city indices (0-indexed)
    """
    tour = []
    in_tour_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('TOUR_SECTION'):
                in_tour_section = True
            elif line == '-1' or line == 'EOF':
                break
            elif in_tour_section:
                try:
                    idx = int(line)
                    if idx > 0:  # TSPLIB uses 1-indexed
                        tour.append(idx - 1)
                except ValueError:
                    pass
    
    return np.array(tour, dtype=np.int64)


def calculate_tour_length(coords: np.ndarray, tour: np.ndarray,
                         edge_weight_type: str = 'EUC_2D') -> float:
    """Calculate tour length according to TSPLIB edge weight type.
    
    Args:
        coords: City coordinates
        tour: Tour as array of city indices
        edge_weight_type: TSPLIB edge weight type
        
    Returns:
        Tour length
    """
    n = len(tour)
    total = 0.0
    
    for i in range(n):
        c1 = coords[tour[i]]
        c2 = coords[tour[(i + 1) % n]]
        
        if edge_weight_type == 'EUC_2D':
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        elif edge_weight_type == 'CEIL_2D':
            dist = np.ceil(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))
        elif edge_weight_type == 'ATT':
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            r = np.sqrt((dx * dx + dy * dy) / 10.0)
            t = np.round(r)
            dist = t + 1 if t < r else t
        elif edge_weight_type == 'GEO':
            # Geographic distance
            lat1 = np.pi * (int(c1[0]) + 5.0 * (c1[0] - int(c1[0])) / 3.0) / 180.0
            lon1 = np.pi * (int(c1[1]) + 5.0 * (c1[1] - int(c1[1])) / 3.0) / 180.0
            lat2 = np.pi * (int(c2[0]) + 5.0 * (c2[0] - int(c2[0])) / 3.0) / 180.0
            lon2 = np.pi * (int(c2[1]) + 5.0 * (c2[1] - int(c2[1])) / 3.0) / 180.0
            
            RRR = 6378.388
            q1 = np.cos(lon1 - lon2)
            q2 = np.cos(lat1 - lat2)
            q3 = np.cos(lat1 + lat2)
            dist = int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        else:
            # Default to Euclidean
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        total += dist
    
    return total


def normalize_coords(coords: np.ndarray, 
                     target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Normalize coordinates to target range.
    
    Args:
        coords: Original coordinates
        target_range: (min, max) for normalized coordinates
        
    Returns:
        Normalized coordinates
    """
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    normalized = (coords - min_vals) / range_vals
    
    # Scale to target range
    target_min, target_max = target_range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


# =============================================================================
# Built-in Benchmark Instances
# =============================================================================

def create_berlin52() -> TSPInstance:
    """Create the berlin52 benchmark instance.
    
    52 locations in Berlin (West Germany).
    Optimal tour length: 7542
    """
    coords = np.array([
        [565, 575], [25, 185], [345, 750], [945, 685], [845, 655],
        [880, 660], [25, 230], [525, 1000], [580, 1175], [650, 1130],
        [1605, 620], [1220, 580], [1465, 200], [1530, 5], [845, 680],
        [725, 370], [145, 665], [415, 635], [510, 875], [560, 365],
        [300, 465], [520, 585], [480, 415], [835, 625], [975, 580],
        [1215, 245], [1320, 315], [1250, 400], [660, 180], [410, 250],
        [420, 555], [575, 665], [1150, 1160], [700, 580], [685, 595],
        [685, 610], [770, 610], [795, 645], [720, 635], [760, 650],
        [475, 960], [95, 260], [875, 920], [700, 500], [555, 815],
        [830, 485], [1170, 65], [830, 610], [605, 625], [595, 360],
        [1340, 725], [1740, 245]
    ], dtype=np.float64)
    
    return TSPInstance(
        name='berlin52',
        comment='52 locations in Berlin (Groetschel)',
        dimension=52,
        coords=coords,
        edge_weight_type='EUC_2D',
        optimal_length=7542
    )


def create_att48() -> TSPInstance:
    """Create the att48 benchmark instance.
    
    48 capitals of the contiguous US states.
    Optimal tour length: 10628
    """
    coords = np.array([
        [6734, 1453], [2233, 10], [5530, 1424], [401, 841], [3082, 1644],
        [7608, 4458], [7573, 3716], [7265, 1268], [6898, 1885], [1112, 2049],
        [5468, 2606], [5989, 2873], [4706, 2674], [4612, 2035], [6347, 2683],
        [6107, 669], [7611, 5184], [7462, 3590], [7732, 4723], [5900, 3561],
        [4483, 3369], [6101, 1110], [5199, 2182], [1633, 2809], [4307, 2322],
        [675, 1006], [7555, 4819], [7541, 3981], [3177, 756], [7352, 4506],
        [7545, 2801], [3245, 3305], [6426, 3173], [4608, 1198], [23, 2216],
        [7248, 3779], [7762, 4595], [7392, 2244], [3484, 2829], [6271, 2135],
        [4985, 140], [1916, 1569], [7280, 4899], [7509, 3239], [10, 2676],
        [6807, 2993], [5185, 3258], [3023, 1942]
    ], dtype=np.float64)
    
    return TSPInstance(
        name='att48',
        comment='48 capitals of the contiguous US states',
        dimension=48,
        coords=coords,
        edge_weight_type='ATT',
        optimal_length=10628
    )


def create_eil51() -> TSPInstance:
    """Create the eil51 benchmark instance.
    
    51 city problem (Christofides/Eilon).
    Optimal tour length: 426
    """
    coords = np.array([
        [37, 52], [49, 49], [52, 64], [20, 26], [40, 30],
        [21, 47], [17, 63], [31, 62], [52, 33], [51, 21],
        [42, 41], [31, 32], [5, 25], [12, 42], [36, 16],
        [52, 41], [27, 23], [17, 33], [13, 13], [57, 58],
        [62, 42], [42, 57], [16, 57], [8, 52], [7, 38],
        [27, 68], [30, 48], [43, 67], [58, 48], [58, 27],
        [37, 69], [38, 46], [46, 10], [61, 33], [62, 63],
        [63, 69], [32, 22], [45, 35], [59, 15], [5, 6],
        [10, 17], [21, 10], [5, 64], [30, 15], [39, 10],
        [32, 39], [25, 32], [25, 55], [48, 28], [56, 37],
        [30, 40]
    ], dtype=np.float64)
    
    return TSPInstance(
        name='eil51',
        comment='51 city problem (Christofides/Eilon)',
        dimension=51,
        coords=coords,
        edge_weight_type='EUC_2D',
        optimal_length=426
    )


# =============================================================================
# TSPLIB Manager
# =============================================================================

class TSPLIBManager:
    """Manager for TSPLIB instances and benchmarks."""
    
    def __init__(self, tsplib_dir: Optional[str] = None):
        """Initialize manager.
        
        Args:
            tsplib_dir: Directory containing TSPLIB files
        """
        self.tsplib_dir = tsplib_dir or os.path.join('data', 'tsplib')
        self._cache: Dict[str, TSPInstance] = {}
        
        # Built-in instances
        self._builtins = {
            'berlin52': create_berlin52,
            'att48': create_att48,
            'eil51': create_eil51,
        }
    
    def get_instance(self, name: str, normalize: bool = True) -> TSPInstance:
        """Get a TSPLIB instance by name.
        
        Args:
            name: Instance name (e.g., 'berlin52')
            normalize: Whether to normalize coordinates to [0, 1]
            
        Returns:
            TSPInstance
        """
        cache_key = f"{name}_{normalize}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check built-ins first
        if name.lower() in self._builtins:
            instance = self._builtins[name.lower()]()
        else:
            # Try to load from file
            filepath = os.path.join(self.tsplib_dir, f"{name}.tsp")
            if not os.path.exists(filepath):
                # Try without extension
                filepath = os.path.join(self.tsplib_dir, name)
            
            if os.path.exists(filepath):
                instance = parse_tsplib(filepath)
            else:
                raise FileNotFoundError(f"TSPLIB instance not found: {name}")
        
        if normalize:
            instance.coords = normalize_coords(instance.coords)
        
        self._cache[cache_key] = instance
        return instance
    
    def list_instances(self) -> List[str]:
        """List available TSPLIB instances."""
        instances = list(self._builtins.keys())
        
        if os.path.exists(self.tsplib_dir):
            for filename in os.listdir(self.tsplib_dir):
                if filename.endswith('.tsp'):
                    instances.append(filename[:-4])
        
        return sorted(set(instances))
    
    def get_optimal_length(self, name: str) -> Optional[float]:
        """Get known optimal length for an instance."""
        return OPTIMAL_SOLUTIONS.get(name.lower())
    
    def benchmark(self, name: str, tour: np.ndarray) -> Dict:
        """Benchmark a tour against known optimal.
        
        Args:
            name: Instance name
            tour: Tour to benchmark
            
        Returns:
            Dictionary with length, optimal, and gap percentage
        """
        instance = self.get_instance(name, normalize=False)
        length = calculate_tour_length(
            instance.coords, tour, instance.edge_weight_type)
        
        result = {
            'name': name,
            'length': length,
            'optimal': instance.optimal_length,
        }
        
        if instance.optimal_length:
            gap = 100 * (length - instance.optimal_length) / instance.optimal_length
            result['gap_percent'] = gap
        
        return result
