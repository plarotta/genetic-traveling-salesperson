"""Main application window for TSP Genetic Algorithm Visualizer."""

import numpy as np
import sys
from typing import Optional, Dict
from threading import Thread
import time

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QSplitter, QMenuBar, QMenu, QFileDialog, QMessageBox,
        QStatusBar, QLabel, QComboBox, QDialog, QDialogButtonBox,
        QFormLayout, QSpinBox, QScrollArea
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QAction, QKeySequence
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QSplitter, QMenuBar, QMenu, QFileDialog, QMessageBox,
        QStatusBar, QLabel, QComboBox, QDialog, QDialogButtonBox,
        QFormLayout, QSpinBox, QScrollArea
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import QAction
    PYQT_VERSION = 5

from gene_tsp.gui.canvas import TSPCanvas
from gene_tsp.gui.controls import ControlPanel
from gene_tsp.gui.charts import StatsDashboard
from gene_tsp.gui.themes import ThemeManager, DARK_THEME, LIGHT_THEME


class WorkerSignals(QObject):
    """Signals for background worker thread."""
    progress = pyqtSignal(int, int, float, float)  # gen, total, best, diversity
    finished = pyqtSignal(object, float)  # best_tour, best_fitness
    error = pyqtSignal(str)
    tour_update = pyqtSignal(object)  # current best tour


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("TSP Genetic Algorithm Visualizer")
        self.setMinimumSize(1200, 800)
        
        # State
        self.coords: Optional[np.ndarray] = None
        self.running = False
        self.paused = False
        self.worker_thread: Optional[Thread] = None
        self.signals = WorkerSignals()
        
        # Theme
        self.theme_manager = ThemeManager()
        
        # Setup UI
        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()
        self._connect_signals()
        
        # Apply theme
        self._apply_theme()
        
        # Load default dataset
        self._load_default_dataset()
    
    def _setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_instance)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Dataset...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_dataset)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Dataset...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_dataset)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Tour...", self)
        export_action.triggered.connect(self._export_tour)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Dataset menu
        dataset_menu = menubar.addMenu("Dataset")
        
        generate_action = QAction("Generate Random...", self)
        generate_action.triggered.connect(self._generate_dataset)
        dataset_menu.addAction(generate_action)
        
        dataset_menu.addSeparator()
        
        # Built-in datasets
        for name in ['Easy (9 cities)', 'Medium (49 cities)', 
                     'Hard (500 cities)', 'Challenge (1000 cities)']:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, n=name: self._load_builtin(n))
            dataset_menu.addAction(action)
        
        dataset_menu.addSeparator()
        
        # TSPLIB instances
        tsplib_menu = dataset_menu.addMenu("TSPLIB")
        for name in ['berlin52', 'eil51', 'att48']:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, n=name: self._load_tsplib(n))
            tsplib_menu.addAction(action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        self.dark_theme_action = QAction("Dark Theme", self, checkable=True)
        self.dark_theme_action.setChecked(True)
        self.dark_theme_action.triggered.connect(lambda: self._set_theme('dark'))
        view_menu.addAction(self.dark_theme_action)
        
        self.light_theme_action = QAction("Light Theme", self, checkable=True)
        self.light_theme_action.triggered.connect(lambda: self._set_theme('light'))
        view_menu.addAction(self.light_theme_action)
        
        view_menu.addSeparator()
        
        self.show_indices_action = QAction("Show City Indices", self, checkable=True)
        self.show_indices_action.triggered.connect(self._toggle_indices)
        view_menu.addAction(self.show_indices_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_ui(self):
        """Build main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Canvas
        self.canvas = TSPCanvas(theme=self.theme_manager.get_theme())
        self.canvas.setMinimumWidth(400)
        splitter.addWidget(self.canvas)
        
        # Right side: Controls and Stats in vertical splitter
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setMinimumWidth(380)
        
        # Control panel in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(300)
        
        self.controls = ControlPanel()
        self.controls.setMinimumWidth(360)
        scroll.setWidget(self.controls)
        right_splitter.addWidget(scroll)
        
        # Stats dashboard
        self.stats = StatsDashboard(theme=self.theme_manager.get_theme())
        self.stats.setMinimumHeight(250)
        right_splitter.addWidget(self.stats)
        
        right_splitter.setSizes([450, 350])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        
        splitter.addWidget(right_splitter)
        splitter.setSizes([650, 400])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def _setup_statusbar(self):
        """Create status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.cities_label = QLabel("Cities: 0")
        self.statusbar.addWidget(self.cities_label)
        
        self.best_label = QLabel("Best: --")
        self.statusbar.addPermanentWidget(self.best_label)
    
    def _connect_signals(self):
        """Connect UI signals."""
        # Control panel
        self.controls.run_clicked.connect(self._start_evolution)
        self.controls.pause_clicked.connect(self._pause_evolution)
        self.controls.stop_clicked.connect(self._stop_evolution)
        self.controls.step_clicked.connect(self._step_evolution)
        self.controls.reset_clicked.connect(self._reset)
        
        # Canvas
        self.canvas.cities_changed.connect(self._on_cities_changed)
        
        # Worker signals
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.tour_update.connect(self._on_tour_update)
        self.signals.error.connect(self._on_error)
        
        # Pareto chart click
        self.stats.pareto_chart.point_clicked.connect(self._on_pareto_click)
    
    def _apply_theme(self):
        """Apply current theme to all widgets."""
        self.setStyleSheet(self.theme_manager.get_stylesheet())
        theme = self.theme_manager.get_theme()
        self.canvas.set_theme(theme)
        self.stats.set_theme(theme)
    
    def _set_theme(self, name: str):
        """Switch theme."""
        self.theme_manager.set_theme(name)
        self._apply_theme()
        
        self.dark_theme_action.setChecked(name == 'dark')
        self.light_theme_action.setChecked(name == 'light')
    
    def _toggle_indices(self, checked: bool):
        """Toggle city index display."""
        self.canvas.show_indices = checked
        self.canvas.redraw()
    
    # =========================================================================
    # Dataset operations
    # =========================================================================
    
    def _load_default_dataset(self):
        """Load a default dataset on startup."""
        try:
            from data.generators import generate_clustered
            coords = generate_clustered(50, n_clusters=5, seed=42)
            self._set_coords(coords)
        except Exception:
            # Fallback to simple random
            coords = np.random.rand(30, 2)
            self._set_coords(coords)
    
    def _set_coords(self, coords: np.ndarray):
        """Set city coordinates."""
        self.coords = coords
        self.canvas.set_cities(coords)
        self.cities_label.setText(f"Cities: {len(coords)}")
        self.stats.clear_all()
    
    def _new_instance(self):
        """Create new empty instance."""
        self.coords = np.empty((0, 2))
        self.canvas.set_cities(self.coords)
        self.stats.clear_all()
        self.cities_label.setText("Cities: 0")
        self.statusbar.showMessage("Click on canvas to add cities", 3000)
    
    def _open_dataset(self):
        """Open dataset from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", "",
            "Text files (*.txt);;TSP files (*.tsp);;All files (*)"
        )
        if filepath:
            try:
                from data.generators import load_dataset
                coords = load_dataset(filepath)
                self._set_coords(coords)
                self.statusbar.showMessage(f"Loaded {len(coords)} cities", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")
    
    def _save_dataset(self):
        """Save current dataset to file."""
        if self.coords is None or len(self.coords) == 0:
            QMessageBox.warning(self, "Warning", "No cities to save")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset", "",
            "Text files (*.txt);;All files (*)"
        )
        if filepath:
            try:
                from data.generators import save_dataset
                save_dataset(self.coords, filepath)
                self.statusbar.showMessage(f"Saved {len(self.coords)} cities", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save dataset: {e}")
    
    def _export_tour(self):
        """Export current best tour."""
        if self.canvas.best_tour is None:
            QMessageBox.warning(self, "Warning", "No tour to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Tour", "",
            "Text files (*.txt);;All files (*)"
        )
        if filepath:
            np.savetxt(filepath, self.canvas.best_tour, fmt='%d')
            self.statusbar.showMessage("Tour exported", 3000)
    
    def _generate_dataset(self):
        """Show dialog to generate dataset."""
        dialog = GenerateDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            try:
                from data.generators import generate_custom
                coords = generate_custom(
                    params['n_cities'],
                    distribution=params['distribution'],
                    seed=params['seed']
                )
                self._set_coords(coords)
                self.statusbar.showMessage(
                    f"Generated {params['n_cities']} cities ({params['distribution']})", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate: {e}")
    
    def _load_builtin(self, name: str):
        """Load built-in dataset."""
        try:
            import os
            name_map = {
                'Easy (9 cities)': 'easy',
                'Medium (49 cities)': 'medium',
                'Hard (500 cities)': 'hard',
                'Challenge (1000 cities)': 'challenge'
            }
            filename = name_map.get(name, 'easy')
            filepath = os.path.join('data', f'{filename}.txt')
            
            delimiter = ',' if filename == 'challenge' else ' '
            coords = np.loadtxt(filepath, delimiter=delimiter)
            self._set_coords(coords)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def _load_tsplib(self, name: str):
        """Load TSPLIB instance."""
        try:
            from gene_tsp.tsplib import TSPLIBManager
            manager = TSPLIBManager()
            instance = manager.get_instance(name, normalize=True)
            self._set_coords(instance.coords)
            
            if instance.optimal_length:
                self.statusbar.showMessage(
                    f"Loaded {name} ({instance.dimension} cities, optimal: {instance.optimal_length})", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def _on_cities_changed(self):
        """Handle cities changed from canvas interaction."""
        self.coords = self.canvas.get_coords()
        n = len(self.coords) if self.coords is not None else 0
        self.cities_label.setText(f"Cities: {n}")
    
    # =========================================================================
    # Evolution control
    # =========================================================================
    
    def _start_evolution(self):
        """Start the evolution process."""
        if self.coords is None or len(self.coords) < 3:
            QMessageBox.warning(self, "Warning", "Need at least 3 cities")
            return
        
        self.running = True
        self.paused = False
        self.controls.set_running(True)
        self.canvas.set_editable(False)
        self.stats.clear_all()
        
        # Get parameters
        params = self.controls.get_params()
        
        # Start worker thread
        self.worker_thread = Thread(
            target=self._evolution_worker,
            args=(self.coords.copy(), params),
            daemon=True
        )
        self.worker_thread.start()
    
    def _pause_evolution(self):
        """Pause/resume evolution."""
        self.paused = not self.paused
        status = "Paused" if self.paused else "Running"
        self.statusbar.showMessage(status, 2000)
    
    def _stop_evolution(self):
        """Stop evolution."""
        self.running = False
        self.paused = False
    
    def _step_evolution(self):
        """Run single generation."""
        # TODO: Implement single step
        self.statusbar.showMessage("Step not yet implemented", 2000)
    
    def _reset(self):
        """Reset to initial state."""
        self._stop_evolution()
        self.canvas.tour = None
        self.canvas.best_tour = None
        self.canvas.redraw()
        self.stats.clear_all()
        self.controls.reset_progress()
        self.controls.set_running(False)
        self.canvas.set_editable(True)
        self.best_label.setText("Best: --")
    
    def _evolution_worker(self, coords: np.ndarray, params: Dict):
        """Worker function for evolution (runs in separate thread)."""
        try:
            algorithm = params.get('algorithm', 'standard')
            
            if algorithm == 'nsga2':
                self._run_nsga2(coords, params)
            elif algorithm == 'island':
                self._run_island_model(coords, params)
            else:
                self._run_standard_ga(coords, params)
                
        except Exception as e:
            self.signals.error.emit(str(e))
    
    def _run_standard_ga(self, coords: np.ndarray, params: Dict):
        """Run standard or memetic GA."""
        from gene_tsp.distance import DistanceMatrix
        from gene_tsp.crossover import get_crossover_operator
        from gene_tsp.mutation import get_mutation_operator, compound_mutation
        from gene_tsp.selection import (tournament_selection_batch, get_elite,
                                        survival_selection, calculate_diversity)
        from gene_tsp.local_search import two_opt
        
        # Setup
        dist_matrix = DistanceMatrix(coords, use_gpu=params.get('use_gpu', False))
        n_cities = len(coords)
        pop_size = params['population_size']
        n_gen = params['generations']
        
        crossover_func = get_crossover_operator(params['crossover'])
        if params['mutation'] == 'compound':
            mutation_func = None  # Use compound_mutation directly
        else:
            mutation_func = get_mutation_operator(params['mutation'])
        
        # Initialize population
        population = np.array([np.random.permutation(n_cities) for _ in range(pop_size)])
        fitness = dist_matrix.get_tour_lengths_batch(population)
        
        # Sort
        indices = np.argsort(fitness)
        population = population[indices]
        fitness = fitness[indices]
        
        best_tour = population[0].copy()
        best_fitness = fitness[0]
        
        for gen in range(n_gen):
            if not self.running:
                break
            
            while self.paused:
                time.sleep(0.1)
                if not self.running:
                    break
            
            # Selection
            parent_indices = tournament_selection_batch(
                fitness, pop_size, params['tournament_size'])
            
            # Create offspring
            offspring = []
            for i in range(0, pop_size, 2):
                p1 = population[parent_indices[i]]
                p2 = population[parent_indices[min(i+1, pop_size-1)]]
                
                # Crossover
                if np.random.random() < params['crossover_rate']:
                    c1, c2 = crossover_func(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                if mutation_func:
                    if np.random.random() < params['mutation_rate']:
                        c1 = mutation_func(c1)
                    if np.random.random() < params['mutation_rate']:
                        c2 = mutation_func(c2)
                else:
                    c1 = compound_mutation(c1, params['mutation_rate'])
                    c2 = compound_mutation(c2, params['mutation_rate'])
                
                # Local search (memetic)
                if params.get('use_local_search'):
                    if np.random.random() < 0.1:
                        c1, _ = two_opt(c1, dist_matrix.matrix, max_iterations=5)
                    if np.random.random() < 0.1:
                        c2, _ = two_opt(c2, dist_matrix.matrix, max_iterations=5)
                
                offspring.extend([c1, c2])
            
            offspring = np.array(offspring[:pop_size])
            offspring_fitness = dist_matrix.get_tour_lengths_batch(offspring)
            
            # Survival selection
            population, fitness = survival_selection(
                population, fitness, offspring, offspring_fitness,
                pop_size, params['elite_size'], method='elitist'
            )
            
            # Sort
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]
            
            # Track best
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_tour = population[0].copy()
            
            # Report progress
            diversity = calculate_diversity(population)
            self.signals.progress.emit(gen + 1, n_gen, best_fitness, diversity)
            
            # Update tour display periodically
            if gen % 5 == 0:
                self.signals.tour_update.emit(best_tour.copy())
            
            # Update stats
            self.stats.update_fitness(best_fitness, np.mean(fitness), np.max(fitness))
            self.stats.update_diversity(diversity)
        
        self.signals.finished.emit(best_tour, best_fitness)
    
    def _run_island_model(self, coords: np.ndarray, params: Dict):
        """Run island model GA."""
        from gene_tsp.island_model import IslandModelGA
        
        model = IslandModelGA(
            coords,
            n_islands=4,
            use_multiprocessing=True
        )
        
        def callback(m):
            if not self.running:
                return
            stats = m.get_statistics()
            self.signals.progress.emit(
                m.generation, params['generations'],
                stats['global_best_fitness'],
                np.mean(stats['island_diversity'])
            )
            if m.generation % 5 == 0:
                self.signals.tour_update.emit(m.global_best_tour.copy())
        
        best_tour, best_fitness = model.evolve(
            params['generations'],
            callback=callback,
            verbose=False
        )
        
        self.signals.finished.emit(best_tour, best_fitness)
    
    def _run_nsga2(self, coords: np.ndarray, params: Dict):
        """Run NSGA-II multi-objective optimization."""
        from gene_tsp.nsga2 import NSGA2, NSGA2Config
        
        config = NSGA2Config(
            population_size=params['population_size'],
            n_generations=params['generations'],
            crossover_rate=params['crossover_rate'],
            mutation_rate=params['mutation_rate'],
        )
        
        nsga2 = NSGA2(coords, objectives=['distance', 'longest_edge'], config=config)
        
        for gen in range(params['generations']):
            if not self.running:
                break
            
            while self.paused:
                time.sleep(0.1)
                if not self.running:
                    break
            
            # One generation
            offspring, offspring_obj = nsga2._create_offspring()
            combined_pop = np.vstack([nsga2.population, offspring])
            combined_obj = np.vstack([nsga2.objectives, offspring_obj])
            nsga2.population, nsga2.objectives = nsga2._survivor_selection(
                combined_pop, combined_obj)
            nsga2._assign_ranks_and_distances()
            
            # Get Pareto front
            pareto_solutions, pareto_objectives = nsga2.get_pareto_front()
            
            # Update UI
            best_distance = np.min(nsga2.objectives[:, 0])
            self.signals.progress.emit(gen + 1, params['generations'], best_distance, 0.0)
            
            if gen % 10 == 0:
                # Update Pareto chart
                self.stats.update_pareto(pareto_objectives, ['Distance', 'Longest Edge'])
                # Show best distance solution
                best_idx = np.argmin(pareto_objectives[:, 0])
                self.signals.tour_update.emit(pareto_solutions[best_idx].copy())
        
        # Final results
        results = nsga2.get_results()
        compromise_tour, compromise_obj = nsga2.get_compromise_solution()
        
        self.stats.show_pareto_tab()
        self.stats.update_pareto(
            results['pareto_objectives'],
            ['Distance', 'Longest Edge']
        )
        
        self.signals.finished.emit(compromise_tour, compromise_obj[0])
    
    def _on_progress(self, gen: int, total: int, best: float, diversity: float):
        """Handle progress update from worker."""
        self.controls.set_progress(gen, total, best, diversity)
        self.best_label.setText(f"Best: {best:.2f}")
    
    def _on_tour_update(self, tour: np.ndarray):
        """Handle tour update from worker."""
        self.canvas.set_best_tour(tour)
    
    def _on_finished(self, best_tour: np.ndarray, best_fitness: float):
        """Handle evolution finished."""
        self.running = False
        self.controls.set_running(False)
        self.canvas.set_editable(True)
        self.canvas.set_best_tour(best_tour)
        self.best_label.setText(f"Best: {best_fitness:.2f}")
        self.statusbar.showMessage(f"Finished! Best tour: {best_fitness:.2f}", 5000)
    
    def _on_error(self, error: str):
        """Handle error from worker."""
        self.running = False
        self.controls.set_running(False)
        self.canvas.set_editable(True)
        QMessageBox.critical(self, "Error", f"Evolution failed: {error}")
    
    def _on_pareto_click(self, idx: int):
        """Handle click on Pareto front point."""
        # This would need access to the Pareto solutions
        # For now, just show a message
        self.statusbar.showMessage(f"Selected Pareto solution {idx}", 2000)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About TSP GA Visualizer",
            "<h3>TSP Genetic Algorithm Visualizer</h3>"
            "<p>An interactive tool for exploring genetic algorithms "
            "applied to the Traveling Salesman Problem.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multiple crossover operators (OX, PMX, ERX)</li>"
            "<li>Local search (2-opt, 3-opt)</li>"
            "<li>Island model parallel GA</li>"
            "<li>NSGA-II multi-objective optimization</li>"
            "<li>GPU acceleration support</li>"
            "</ul>"
            "<p>Built with PyQt6 and Matplotlib</p>"
        )


class GenerateDialog(QDialog):
    """Dialog for generating random datasets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Dataset")
        self.setMinimumWidth(300)
        
        layout = QFormLayout(self)
        
        self.n_cities_spin = QSpinBox()
        self.n_cities_spin.setRange(3, 10000)
        self.n_cities_spin.setValue(100)
        layout.addRow("Number of cities:", self.n_cities_spin)
        
        self.dist_combo = QComboBox()
        self.dist_combo.addItems([
            'uniform', 'clustered', 'circle', 'grid',
            'star', 'concentric', 'two_clusters'
        ])
        self.dist_combo.setCurrentText('clustered')
        layout.addRow("Distribution:", self.dist_combo)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        layout.addRow("Random seed:", self.seed_spin)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_params(self) -> Dict:
        return {
            'n_cities': self.n_cities_spin.value(),
            'distribution': self.dist_combo.currentText(),
            'seed': self.seed_spin.value(),
        }


def run_app():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("TSP GA Visualizer")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()
