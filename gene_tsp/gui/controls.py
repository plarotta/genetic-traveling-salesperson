"""Control panel for algorithm parameters and execution.

Provides sliders, dropdowns, and buttons for configuring and running the GA.
"""

import numpy as np
from typing import Optional, Callable, Dict

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
        QDoubleSpinBox, QGroupBox, QCheckBox, QProgressBar,
        QFrame
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
        QDoubleSpinBox, QGroupBox, QCheckBox, QProgressBar,
        QFrame
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    PYQT_VERSION = 5


class LabeledSlider(QWidget):
    """Slider with label and value display."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, decimals: int = 2, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.multiplier = 10 ** decimals
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
        
        self.label = QLabel(label)
        self.label.setMinimumWidth(110)
        self.label.setStyleSheet("font-size: 13px;")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
        self.slider.setMinimumHeight(22)
        self.slider.valueChanged.connect(self._on_change)
        
        self.value_label = QLabel(f"{default:.{decimals}f}")
        self.value_label.setMinimumWidth(45)
        self.value_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        
        self.setMinimumHeight(30)
    
    def _on_change(self, value: int):
        float_val = value / self.multiplier
        self.value_label.setText(f"{float_val:.{self.decimals}f}")
        self.valueChanged.emit(float_val)
    
    def value(self) -> float:
        return self.slider.value() / self.multiplier
    
    def setValue(self, val: float):
        self.slider.setValue(int(val * self.multiplier))


class ControlPanel(QWidget):
    """Main control panel for algorithm configuration.
    
    Signals:
        run_clicked: Start button pressed
        pause_clicked: Pause button pressed
        stop_clicked: Stop button pressed
        step_clicked: Single step button pressed
        reset_clicked: Reset button pressed
        params_changed: Any parameter changed
    """
    
    run_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    params_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Execution controls
        exec_group = QGroupBox("Execution")
        exec_group.setMinimumHeight(70)
        exec_layout = QHBoxLayout(exec_group)
        exec_layout.setSpacing(8)
        
        btn_style = "min-height: 32px; min-width: 70px; font-size: 13px; font-weight: bold;"
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet(btn_style)
        self.run_btn.clicked.connect(self.run_clicked.emit)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setStyleSheet(btn_style)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.pause_btn.setEnabled(False)
        
        self.step_btn = QPushButton("⏭ Step")
        self.step_btn.setStyleSheet(btn_style)
        self.step_btn.clicked.connect(self.step_clicked.emit)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(btn_style)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setProperty("class", "danger")
        
        exec_layout.addWidget(self.run_btn)
        exec_layout.addWidget(self.pause_btn)
        exec_layout.addWidget(self.step_btn)
        exec_layout.addWidget(self.stop_btn)
        
        layout.addWidget(exec_group)
        
        # Algorithm selection
        algo_group = QGroupBox("Algorithm")
        algo_group.setMinimumHeight(130)
        algo_layout = QGridLayout(algo_group)
        algo_layout.setSpacing(8)
        algo_layout.setColumnStretch(1, 1)
        
        combo_style = "min-height: 28px; font-size: 13px; padding: 4px 8px;"
        label_style = "font-size: 13px;"
        
        method_label = QLabel("Method:")
        method_label.setStyleSheet(label_style)
        algo_layout.addWidget(method_label, 0, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.setStyleSheet(combo_style)
        self.algo_combo.addItems([
            "Standard GA",
            "Memetic GA (GA + 2-opt)",
            "Island Model",
            "NSGA-II (Multi-objective)"
        ])
        self.algo_combo.currentIndexChanged.connect(self._on_param_change)
        algo_layout.addWidget(self.algo_combo, 0, 1)
        
        cross_label = QLabel("Crossover:")
        cross_label.setStyleSheet(label_style)
        algo_layout.addWidget(cross_label, 1, 0)
        self.crossover_combo = QComboBox()
        self.crossover_combo.setStyleSheet(combo_style)
        self.crossover_combo.addItems(["Order (OX)", "PMX", "Edge Recombination"])
        self.crossover_combo.currentIndexChanged.connect(self._on_param_change)
        algo_layout.addWidget(self.crossover_combo, 1, 1)
        
        mut_label = QLabel("Mutation:")
        mut_label.setStyleSheet(label_style)
        algo_layout.addWidget(mut_label, 2, 0)
        self.mutation_combo = QComboBox()
        self.mutation_combo.setStyleSheet(combo_style)
        self.mutation_combo.addItems(["Inversion", "Swap", "Insert", "Scramble", "Compound"])
        self.mutation_combo.currentIndexChanged.connect(self._on_param_change)
        algo_layout.addWidget(self.mutation_combo, 2, 1)
        
        layout.addWidget(algo_group)
        
        # Population parameters
        pop_group = QGroupBox("Population")
        pop_group.setMinimumHeight(90)
        pop_layout = QVBoxLayout(pop_group)
        pop_layout.setSpacing(8)
        
        spin_style = "min-height: 26px; font-size: 13px; padding: 2px 6px;"
        
        # Population size
        pop_size_layout = QHBoxLayout()
        pop_size_label = QLabel("Population Size:")
        pop_size_label.setStyleSheet(label_style)
        pop_size_layout.addWidget(pop_size_label)
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setStyleSheet(spin_style)
        self.pop_size_spin.setRange(10, 1000)
        self.pop_size_spin.setValue(100)
        self.pop_size_spin.valueChanged.connect(self._on_param_change)
        pop_size_layout.addWidget(self.pop_size_spin)
        pop_layout.addLayout(pop_size_layout)
        
        # Generations
        gen_layout = QHBoxLayout()
        gen_label = QLabel("Generations:")
        gen_label.setStyleSheet(label_style)
        gen_layout.addWidget(gen_label)
        self.generations_spin = QSpinBox()
        self.generations_spin.setStyleSheet(spin_style)
        self.generations_spin.setRange(10, 100000)
        self.generations_spin.setValue(500)
        self.generations_spin.valueChanged.connect(self._on_param_change)
        gen_layout.addWidget(self.generations_spin)
        pop_layout.addLayout(gen_layout)
        
        layout.addWidget(pop_group)
        
        # Genetic parameters
        genetic_group = QGroupBox("Genetic Parameters")
        genetic_group.setMinimumHeight(160)
        genetic_layout = QVBoxLayout(genetic_group)
        genetic_layout.setSpacing(8)
        
        self.crossover_rate = LabeledSlider("Crossover Rate:", 0.0, 1.0, 0.9)
        self.crossover_rate.valueChanged.connect(self._on_param_change)
        genetic_layout.addWidget(self.crossover_rate)
        
        self.mutation_rate = LabeledSlider("Mutation Rate:", 0.0, 1.0, 0.2)
        self.mutation_rate.valueChanged.connect(self._on_param_change)
        genetic_layout.addWidget(self.mutation_rate)
        
        self.elite_size = LabeledSlider("Elite Size:", 0, 20, 2, decimals=0)
        self.elite_size.valueChanged.connect(self._on_param_change)
        genetic_layout.addWidget(self.elite_size)
        
        self.tournament_size = LabeledSlider("Tournament Size:", 2, 10, 3, decimals=0)
        self.tournament_size.valueChanged.connect(self._on_param_change)
        genetic_layout.addWidget(self.tournament_size)
        
        layout.addWidget(genetic_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced")
        advanced_group.setMinimumHeight(110)
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(6)
        
        checkbox_style = "font-size: 13px; min-height: 24px;"
        
        self.use_local_search = QCheckBox("Use Local Search (2-opt)")
        self.use_local_search.setStyleSheet(checkbox_style)
        self.use_local_search.stateChanged.connect(self._on_param_change)
        advanced_layout.addWidget(self.use_local_search)
        
        self.adaptive_mutation = QCheckBox("Adaptive Mutation Rate")
        self.adaptive_mutation.setStyleSheet(checkbox_style)
        self.adaptive_mutation.stateChanged.connect(self._on_param_change)
        advanced_layout.addWidget(self.adaptive_mutation)
        
        self.use_gpu = QCheckBox("GPU Acceleration (if available)")
        self.use_gpu.setStyleSheet(checkbox_style)
        self.use_gpu.stateChanged.connect(self._on_param_change)
        advanced_layout.addWidget(self.use_gpu)
        
        layout.addWidget(advanced_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_group.setMinimumHeight(80)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(6)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(22)
        self.progress_bar.setStyleSheet("font-size: 12px;")
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: #888;")
        self.status_label.setProperty("class", "subheading")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Reset button
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.setStyleSheet("min-height: 32px; font-size: 13px; font-weight: bold;")
        self.reset_btn.setProperty("class", "secondary")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_btn)
    
    def _on_param_change(self, *args):
        """Emit params_changed signal with current parameters."""
        self.params_changed.emit(self.get_params())
    
    def get_params(self) -> Dict:
        """Get all current parameter values."""
        algo_map = {
            0: 'standard',
            1: 'memetic',
            2: 'island',
            3: 'nsga2'
        }
        
        crossover_map = {
            0: 'ox',
            1: 'pmx',
            2: 'erx'
        }
        
        mutation_map = {
            0: 'inversion',
            1: 'swap',
            2: 'insert',
            3: 'scramble',
            4: 'compound'
        }
        
        return {
            'algorithm': algo_map.get(self.algo_combo.currentIndex(), 'standard'),
            'crossover': crossover_map.get(self.crossover_combo.currentIndex(), 'ox'),
            'mutation': mutation_map.get(self.mutation_combo.currentIndex(), 'inversion'),
            'population_size': self.pop_size_spin.value(),
            'generations': self.generations_spin.value(),
            'crossover_rate': self.crossover_rate.value(),
            'mutation_rate': self.mutation_rate.value(),
            'elite_size': int(self.elite_size.value()),
            'tournament_size': int(self.tournament_size.value()),
            'use_local_search': self.use_local_search.isChecked(),
            'adaptive_mutation': self.adaptive_mutation.isChecked(),
            'use_gpu': self.use_gpu.isChecked(),
        }
    
    def set_params(self, params: Dict):
        """Set parameters from dictionary."""
        if 'population_size' in params:
            self.pop_size_spin.setValue(params['population_size'])
        if 'generations' in params:
            self.generations_spin.setValue(params['generations'])
        if 'crossover_rate' in params:
            self.crossover_rate.setValue(params['crossover_rate'])
        if 'mutation_rate' in params:
            self.mutation_rate.setValue(params['mutation_rate'])
        if 'elite_size' in params:
            self.elite_size.setValue(params['elite_size'])
        if 'tournament_size' in params:
            self.tournament_size.setValue(params['tournament_size'])
        if 'use_local_search' in params:
            self.use_local_search.setChecked(params['use_local_search'])
        if 'adaptive_mutation' in params:
            self.adaptive_mutation.setChecked(params['adaptive_mutation'])
        if 'use_gpu' in params:
            self.use_gpu.setChecked(params['use_gpu'])
    
    def set_running(self, running: bool):
        """Update UI for running/stopped state."""
        self.run_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.step_btn.setEnabled(not running)
        
        # Disable parameter changes while running
        self.algo_combo.setEnabled(not running)
        self.crossover_combo.setEnabled(not running)
        self.mutation_combo.setEnabled(not running)
        self.pop_size_spin.setEnabled(not running)
        self.generations_spin.setEnabled(not running)
    
    def set_progress(self, generation: int, total: int, 
                     best_fitness: float, diversity: float = 0.0):
        """Update progress display."""
        if total > 0:
            self.progress_bar.setValue(int(100 * generation / total))
        
        self.status_label.setText(
            f"Gen {generation}/{total} | Best: {best_fitness:.2f} | "
            f"Diversity: {diversity:.3f}"
        )
    
    def reset_progress(self):
        """Reset progress to initial state."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
