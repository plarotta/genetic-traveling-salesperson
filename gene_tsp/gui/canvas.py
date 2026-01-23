"""Interactive canvas for TSP visualization.

Provides city manipulation (add/move/remove) and tour animation.
"""

import numpy as np
from typing import Optional, List, Callable, Tuple
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
    from PyQt6.QtCore import pyqtSignal, Qt
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
    from PyQt5.QtCore import pyqtSignal, Qt
    PYQT_VERSION = 5


class TSPCanvas(QWidget):
    """Interactive canvas for visualizing TSP solutions.
    
    Features:
    - Click to add cities
    - Drag to move cities
    - Right-click to remove cities
    - Animated tour display
    - Edge frequency heatmap
    
    Signals:
        cities_changed: Emitted when cities are added/moved/removed
        city_selected: Emitted when a city is clicked
    """
    
    cities_changed = pyqtSignal()
    city_selected = pyqtSignal(int)
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        
        self.theme = theme
        self.coords: Optional[np.ndarray] = None
        self.tour: Optional[np.ndarray] = None
        self.best_tour: Optional[np.ndarray] = None
        
        # Interaction state
        self.dragging = False
        self.drag_idx = -1
        self.editable = True
        
        # Visual settings
        self.city_size = 80
        self.line_width = 1.5
        self.best_line_width = 2.5
        self.show_grid = True
        self.show_indices = False
        
        # Animation
        self.animation: Optional[FuncAnimation] = None
        self.animation_speed = 50  # ms between frames
        
        self._setup_canvas()
        self._connect_events()
    
    def _setup_canvas(self):
        """Set up matplotlib figure and canvas."""
        self.figure = Figure(figsize=(8, 8), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, 
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(400, 400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        
        self._apply_theme()
        self._setup_axes()
    
    def _apply_theme(self):
        """Apply theme colors to canvas."""
        if self.theme:
            self.figure.set_facecolor(self.theme.canvas_bg)
            self.ax.set_facecolor(self.theme.canvas_bg)
            self.city_color = self.theme.city_color
            self.tour_color = self.theme.tour_color
            self.best_color = self.theme.best_tour_color
            self.grid_color = self.theme.grid_color
            self.text_color = self.theme.text_primary
        else:
            self.city_color = '#22d3ee'
            self.tour_color = '#6366f1'
            self.best_color = '#10b981'
            self.grid_color = '#1e1e2e'
            self.text_color = '#f1f5f9'
    
    def _setup_axes(self):
        """Configure axes appearance."""
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        if self.show_grid:
            self.ax.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
    
    def _connect_events(self):
        """Connect matplotlib events for interaction."""
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
    
    def _on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.ax or not self.editable:
            return
        
        if self.coords is None:
            return
        
        x, y = event.xdata, event.ydata
        
        # Check if clicking on existing city
        distances = np.sqrt(np.sum((self.coords - [x, y])**2, axis=1))
        min_dist_idx = np.argmin(distances)
        click_threshold = 0.05
        
        if distances[min_dist_idx] < click_threshold:
            if event.button == 3:  # Right click - remove
                self.remove_city(min_dist_idx)
            else:  # Left click - start drag
                self.dragging = True
                self.drag_idx = min_dist_idx
                self.city_selected.emit(min_dist_idx)
        elif event.button == 1:  # Left click on empty space - add city
            self.add_city(x, y)
    
    def _on_release(self, event):
        """Handle mouse release."""
        self.dragging = False
        self.drag_idx = -1
    
    def _on_motion(self, event):
        """Handle mouse motion (drag)."""
        if not self.dragging or event.inaxes != self.ax:
            return
        
        if self.drag_idx >= 0 and self.coords is not None:
            # Update city position
            self.coords[self.drag_idx] = [event.xdata, event.ydata]
            self.redraw()
            self.cities_changed.emit()
    
    def set_cities(self, coords: np.ndarray):
        """Set city coordinates.
        
        Args:
            coords: Array of shape (n_cities, 2)
        """
        self.coords = np.asarray(coords, dtype=np.float64).copy()
        self.tour = None
        self.best_tour = None
        self.redraw()
    
    def add_city(self, x: float, y: float):
        """Add a new city."""
        if self.coords is None:
            self.coords = np.array([[x, y]])
        else:
            self.coords = np.vstack([self.coords, [x, y]])
        self.redraw()
        self.cities_changed.emit()
    
    def remove_city(self, idx: int):
        """Remove a city by index."""
        if self.coords is not None and len(self.coords) > 0:
            self.coords = np.delete(self.coords, idx, axis=0)
            self.tour = None
            self.best_tour = None
            self.redraw()
            self.cities_changed.emit()
    
    def clear_cities(self):
        """Remove all cities."""
        self.coords = None
        self.tour = None
        self.best_tour = None
        self.redraw()
        self.cities_changed.emit()
    
    def set_tour(self, tour: np.ndarray):
        """Set current tour to display.
        
        Args:
            tour: Array of city indices
        """
        self.tour = np.asarray(tour, dtype=np.int64)
        self.redraw()
    
    def set_best_tour(self, tour: np.ndarray):
        """Set best tour (displayed differently).
        
        Args:
            tour: Array of city indices
        """
        self.best_tour = np.asarray(tour, dtype=np.int64)
        self.redraw()
    
    def redraw(self):
        """Redraw the canvas."""
        self.ax.clear()
        self._setup_axes()
        
        if self.coords is None or len(self.coords) == 0:
            self.canvas.draw()
            return
        
        # Draw tour edges
        if self.tour is not None:
            self._draw_tour(self.tour, self.tour_color, self.line_width, alpha=0.6)
        
        # Draw best tour (on top)
        if self.best_tour is not None:
            self._draw_tour(self.best_tour, self.best_color, 
                           self.best_line_width, alpha=0.9)
        
        # Draw cities
        self.ax.scatter(
            self.coords[:, 0], self.coords[:, 1],
            s=self.city_size, c=self.city_color,
            edgecolors='white', linewidths=1.5,
            zorder=10
        )
        
        # Draw indices if enabled
        if self.show_indices:
            for i, (x, y) in enumerate(self.coords):
                self.ax.annotate(
                    str(i), (x, y),
                    fontsize=8, color=self.text_color,
                    ha='center', va='center',
                    zorder=11
                )
        
        self.canvas.draw()
    
    def _draw_tour(self, tour: np.ndarray, color: str, 
                   width: float, alpha: float = 1.0):
        """Draw tour edges."""
        if len(tour) < 2:
            return
        
        # Create line segments
        points = self.coords[tour]
        segments = []
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            segments.append([points[i], points[j]])
        
        lc = LineCollection(segments, colors=color, linewidths=width, alpha=alpha)
        self.ax.add_collection(lc)
    
    def animate_tour_evolution(self, tours: List[np.ndarray], 
                               interval: int = 100,
                               callback: Optional[Callable] = None):
        """Animate through a sequence of tours.
        
        Args:
            tours: List of tour arrays
            interval: Milliseconds between frames
            callback: Called with (frame_idx, tour) each frame
        """
        if self.animation:
            self.animation.event_source.stop()
        
        def update(frame):
            self.tour = tours[frame]
            self.redraw()
            if callback:
                callback(frame, tours[frame])
            return []
        
        self.animation = FuncAnimation(
            self.figure, update,
            frames=len(tours),
            interval=interval,
            repeat=False,
            blit=True
        )
        self.canvas.draw()
    
    def stop_animation(self):
        """Stop any running animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def draw_pareto_point(self, solution: np.ndarray, highlight: bool = False):
        """Highlight a solution from the Pareto front.
        
        Args:
            solution: Tour to display
            highlight: Whether to use highlight styling
        """
        if highlight:
            self.best_tour = solution
        else:
            self.tour = solution
        self.redraw()
    
    def draw_edge_heatmap(self, edge_frequencies: np.ndarray):
        """Draw heatmap showing edge usage frequencies.
        
        Args:
            edge_frequencies: Matrix of edge frequencies (n_cities, n_cities)
        """
        if self.coords is None:
            return
        
        self.ax.clear()
        self._setup_axes()
        
        n = len(self.coords)
        max_freq = edge_frequencies.max()
        if max_freq == 0:
            max_freq = 1
        
        # Draw edges with alpha based on frequency
        for i in range(n):
            for j in range(i + 1, n):
                freq = edge_frequencies[i, j] + edge_frequencies[j, i]
                if freq > 0:
                    alpha = 0.1 + 0.9 * (freq / max_freq)
                    self.ax.plot(
                        [self.coords[i, 0], self.coords[j, 0]],
                        [self.coords[i, 1], self.coords[j, 1]],
                        color=self.tour_color, alpha=alpha,
                        linewidth=1 + 2 * (freq / max_freq)
                    )
        
        # Draw cities
        self.ax.scatter(
            self.coords[:, 0], self.coords[:, 1],
            s=self.city_size, c=self.city_color,
            edgecolors='white', linewidths=1.5,
            zorder=10
        )
        
        self.canvas.draw()
    
    def set_theme(self, theme):
        """Update theme."""
        self.theme = theme
        self._apply_theme()
        self.redraw()
    
    def set_editable(self, editable: bool):
        """Enable/disable city editing."""
        self.editable = editable
    
    def get_coords(self) -> Optional[np.ndarray]:
        """Get current city coordinates."""
        return self.coords.copy() if self.coords is not None else None
