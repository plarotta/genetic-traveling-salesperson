"""Statistics dashboard with real-time charts.

Displays fitness history, diversity metrics, and Pareto fronts.
"""

import numpy as np
from typing import Optional, List, Dict
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
        QLabel, QSizePolicy, QScrollArea
    )
    from PyQt6.QtCore import Qt
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
        QLabel, QSizePolicy, QScrollArea
    )
    from PyQt5.QtCore import Qt
    PYQT_VERSION = 5


class FitnessChart(QWidget):
    """Real-time fitness evolution chart."""
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        
        self.best_history: List[float] = []
        self.avg_history: List[float] = []
        self.worst_history: List[float] = []
        
        self._setup_chart()
    
    def _setup_chart(self):
        """Set up matplotlib figure."""
        self.figure = Figure(figsize=(8, 4), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme colors."""
        if self.theme:
            self.figure.set_facecolor(self.theme.chart_bg)
            self.ax.set_facecolor(self.theme.chart_bg)
            self.line_colors = [
                self.theme.chart_line1,
                self.theme.chart_line2,
                self.theme.chart_line3
            ]
            self.text_color = self.theme.text_primary
            self.grid_color = self.theme.grid_color
        else:
            self.line_colors = ['#6366f1', '#22d3ee', '#f59e0b']
            self.text_color = '#f1f5f9'
            self.grid_color = '#1e1e2e'
    
    def update(self, best: float, avg: Optional[float] = None,
               worst: Optional[float] = None):
        """Add new data point and redraw.
        
        Args:
            best: Best fitness
            avg: Average fitness (optional)
            worst: Worst fitness (optional)
        """
        self.best_history.append(best)
        if avg is not None:
            self.avg_history.append(avg)
        if worst is not None:
            self.worst_history.append(worst)
        
        self._redraw()
    
    def _redraw(self):
        """Redraw the chart."""
        self.ax.clear()
        
        generations = range(len(self.best_history))
        
        # Plot lines
        self.ax.plot(generations, self.best_history, 
                    color=self.line_colors[0], linewidth=2, label='Best')
        
        if self.avg_history:
            self.ax.plot(generations, self.avg_history,
                        color=self.line_colors[1], linewidth=1.5, 
                        linestyle='--', label='Average')
        
        if self.worst_history:
            self.ax.plot(generations, self.worst_history,
                        color=self.line_colors[2], linewidth=1, 
                        linestyle=':', alpha=0.5, label='Worst')
        
        # Fill area between best and avg
        if self.avg_history:
            self.ax.fill_between(generations, self.best_history, self.avg_history,
                                alpha=0.1, color=self.line_colors[0])
        
        # Styling
        self.ax.set_xlabel('Generation', color=self.text_color, fontsize=10)
        self.ax.set_ylabel('Tour Length', color=self.text_color, fontsize=10)
        self.ax.set_title('Fitness Evolution', color=self.text_color, fontsize=11, fontweight='bold')
        self.ax.tick_params(colors=self.text_color, labelsize=9)
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
        self.ax.legend(loc='upper right', framealpha=0.8, fontsize=9)
        
        # Style spines
        for spine in self.ax.spines.values():
            spine.set_color(self.grid_color)
        
        self.canvas.draw()
    
    def clear(self):
        """Clear all data."""
        self.best_history.clear()
        self.avg_history.clear()
        self.worst_history.clear()
        self.ax.clear()
        self.canvas.draw()
    
    def set_theme(self, theme):
        """Update theme."""
        self.theme = theme
        self._apply_theme()
        self._redraw()


class DiversityChart(QWidget):
    """Population diversity over time."""
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        
        self.diversity_history: List[float] = []
        
        self._setup_chart()
    
    def _setup_chart(self):
        """Set up matplotlib figure."""
        self.figure = Figure(figsize=(8, 3), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme colors."""
        if self.theme:
            self.figure.set_facecolor(self.theme.chart_bg)
            self.ax.set_facecolor(self.theme.chart_bg)
            self.line_color = self.theme.accent
            self.text_color = self.theme.text_primary
            self.grid_color = self.theme.grid_color
        else:
            self.line_color = '#22d3ee'
            self.text_color = '#f1f5f9'
            self.grid_color = '#1e1e2e'
    
    def update(self, diversity: float):
        """Add new diversity value."""
        self.diversity_history.append(diversity)
        self._redraw()
    
    def _redraw(self):
        """Redraw the chart."""
        self.ax.clear()
        
        generations = range(len(self.diversity_history))
        
        self.ax.fill_between(generations, 0, self.diversity_history,
                            alpha=0.3, color=self.line_color)
        self.ax.plot(generations, self.diversity_history,
                    color=self.line_color, linewidth=2)
        
        self.ax.set_xlabel('Generation', color=self.text_color, fontsize=10)
        self.ax.set_ylabel('Diversity', color=self.text_color, fontsize=10)
        self.ax.set_title('Population Diversity', color=self.text_color, fontsize=11, fontweight='bold')
        self.ax.tick_params(colors=self.text_color, labelsize=9)
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
        self.ax.set_ylim(0, max(1, max(self.diversity_history) * 1.1) if self.diversity_history else 1)
        
        for spine in self.ax.spines.values():
            spine.set_color(self.grid_color)
        
        self.canvas.draw()
    
    def clear(self):
        """Clear all data."""
        self.diversity_history.clear()
        self.ax.clear()
        self.canvas.draw()
    
    def set_theme(self, theme):
        """Update theme."""
        self.theme = theme
        self._apply_theme()
        if self.diversity_history:
            self._redraw()


class ParetoChart(QWidget):
    """Pareto front visualization for NSGA-II."""
    
    # Signal emitted when a point is clicked
    try:
        from PyQt6.QtCore import pyqtSignal
        point_clicked = pyqtSignal(int)
    except ImportError:
        from PyQt5.QtCore import pyqtSignal
        point_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        
        self.objectives: Optional[np.ndarray] = None
        self.objective_names = ['Distance', 'Longest Edge']
        self.selected_idx = -1
        
        self._setup_chart()
    
    def _setup_chart(self):
        """Set up matplotlib figure."""
        self.figure = Figure(figsize=(6, 5), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self._on_click)
        
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme colors."""
        if self.theme:
            self.figure.set_facecolor(self.theme.chart_bg)
            self.ax.set_facecolor(self.theme.chart_bg)
            self.point_color = self.theme.primary
            self.selected_color = self.theme.success
            self.text_color = self.theme.text_primary
            self.grid_color = self.theme.grid_color
        else:
            self.point_color = '#6366f1'
            self.selected_color = '#10b981'
            self.text_color = '#f1f5f9'
            self.grid_color = '#1e1e2e'
    
    def update(self, objectives: np.ndarray, 
               objective_names: Optional[List[str]] = None):
        """Update Pareto front display.
        
        Args:
            objectives: Array of shape (n_solutions, 2) with objective values
            objective_names: Names for the two objectives
        """
        self.objectives = objectives
        if objective_names:
            self.objective_names = objective_names
        self._redraw()
    
    def _redraw(self):
        """Redraw the chart."""
        self.ax.clear()
        
        if self.objectives is None or len(self.objectives) == 0:
            self.canvas.draw()
            return
        
        # Plot all points
        colors = [self.selected_color if i == self.selected_idx else self.point_color
                  for i in range(len(self.objectives))]
        sizes = [150 if i == self.selected_idx else 80
                for i in range(len(self.objectives))]
        
        self.scatter = self.ax.scatter(
            self.objectives[:, 0], self.objectives[:, 1],
            c=colors, s=sizes, alpha=0.8,
            edgecolors='white', linewidths=1
        )
        
        # Connect points to show Pareto front
        sorted_idx = np.argsort(self.objectives[:, 0])
        sorted_obj = self.objectives[sorted_idx]
        self.ax.plot(sorted_obj[:, 0], sorted_obj[:, 1],
                    color=self.point_color, alpha=0.4, linestyle='--')
        
        # Labels
        self.ax.set_xlabel(self.objective_names[0], color=self.text_color, fontsize=10)
        self.ax.set_ylabel(self.objective_names[1], color=self.text_color, fontsize=10)
        self.ax.set_title('Pareto Front', color=self.text_color, fontsize=11, fontweight='bold')
        self.ax.tick_params(colors=self.text_color, labelsize=9)
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
        
        # Add annotation for ideal direction
        self.ax.annotate('← Better', xy=(0.95, 0.02), xycoords='axes fraction',
                        color=self.text_color, fontsize=8, alpha=0.7, ha='right')
        self.ax.annotate('↓ Better', xy=(0.02, 0.95), xycoords='axes fraction',
                        color=self.text_color, fontsize=8, alpha=0.7, rotation=90)
        
        for spine in self.ax.spines.values():
            spine.set_color(self.grid_color)
        
        self.canvas.draw()
    
    def _on_click(self, event):
        """Handle click to select point."""
        if event.inaxes != self.ax or self.objectives is None:
            return
        
        # Find nearest point
        distances = np.sqrt(
            (self.objectives[:, 0] - event.xdata)**2 +
            (self.objectives[:, 1] - event.ydata)**2
        )
        
        # Normalize by axis ranges
        x_range = self.objectives[:, 0].max() - self.objectives[:, 0].min()
        y_range = self.objectives[:, 1].max() - self.objectives[:, 1].min()
        if x_range > 0 and y_range > 0:
            norm_distances = np.sqrt(
                ((self.objectives[:, 0] - event.xdata) / x_range)**2 +
                ((self.objectives[:, 1] - event.ydata) / y_range)**2
            )
            if norm_distances.min() < 0.1:  # Click threshold
                self.selected_idx = np.argmin(norm_distances)
                self._redraw()
                self.point_clicked.emit(self.selected_idx)
    
    def clear(self):
        """Clear the chart."""
        self.objectives = None
        self.selected_idx = -1
        self.ax.clear()
        self.canvas.draw()
    
    def set_theme(self, theme):
        """Update theme."""
        self.theme = theme
        self._apply_theme()
        if self.objectives is not None:
            self._redraw()


class StatsDashboard(QWidget):
    """Complete statistics dashboard with multiple charts."""
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Fitness tab
        fitness_widget = QWidget()
        fitness_layout = QVBoxLayout(fitness_widget)
        fitness_layout.setContentsMargins(4, 8, 4, 4)
        self.fitness_chart = FitnessChart(theme=self.theme)
        fitness_layout.addWidget(self.fitness_chart)
        self.tabs.addTab(fitness_widget, "Fitness")
        
        # Diversity tab
        diversity_widget = QWidget()
        diversity_layout = QVBoxLayout(diversity_widget)
        diversity_layout.setContentsMargins(4, 8, 4, 4)
        self.diversity_chart = DiversityChart(theme=self.theme)
        diversity_layout.addWidget(self.diversity_chart)
        self.tabs.addTab(diversity_widget, "Diversity")
        
        # Pareto tab (for NSGA-II)
        pareto_widget = QWidget()
        pareto_layout = QVBoxLayout(pareto_widget)
        pareto_layout.setContentsMargins(4, 8, 4, 4)
        self.pareto_chart = ParetoChart(theme=self.theme)
        pareto_layout.addWidget(self.pareto_chart)
        self.tabs.addTab(pareto_widget, "Pareto Front")
        
        layout.addWidget(self.tabs, 1)  # Give tabs stretch priority
        
        # Stats summary
        self.stats_label = QLabel("Best: -- | Avg: -- | Generation: 0")
        self.stats_label.setProperty("class", "subheading")
        self.stats_label.setMinimumHeight(24)
        layout.addWidget(self.stats_label)
    
    def update_fitness(self, best: float, avg: Optional[float] = None,
                       worst: Optional[float] = None):
        """Update fitness chart."""
        self.fitness_chart.update(best, avg, worst)
    
    def update_diversity(self, diversity: float):
        """Update diversity chart."""
        self.diversity_chart.update(diversity)
    
    def update_pareto(self, objectives: np.ndarray,
                      objective_names: Optional[List[str]] = None):
        """Update Pareto front chart."""
        self.pareto_chart.update(objectives, objective_names)
    
    def update_stats(self, generation: int, best: float, 
                     avg: Optional[float] = None):
        """Update stats summary label."""
        avg_str = f"{avg:.2f}" if avg else "--"
        self.stats_label.setText(
            f"Best: {best:.2f} | Avg: {avg_str} | Generation: {generation}"
        )
    
    def clear_all(self):
        """Clear all charts."""
        self.fitness_chart.clear()
        self.diversity_chart.clear()
        self.pareto_chart.clear()
        self.stats_label.setText("Best: -- | Avg: -- | Generation: 0")
    
    def set_theme(self, theme):
        """Update theme for all charts."""
        self.theme = theme
        self.fitness_chart.set_theme(theme)
        self.diversity_chart.set_theme(theme)
        self.pareto_chart.set_theme(theme)
    
    def show_pareto_tab(self):
        """Switch to Pareto front tab."""
        self.tabs.setCurrentIndex(2)
