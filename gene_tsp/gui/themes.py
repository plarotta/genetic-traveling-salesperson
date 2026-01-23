"""Theme definitions for the TSP GUI.

Provides dark and light themes with consistent color schemes.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Color theme definition."""
    name: str
    
    # Main colors
    background: str
    surface: str
    primary: str
    secondary: str
    accent: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str
    
    # Status colors
    success: str
    warning: str
    error: str
    
    # Canvas colors
    canvas_bg: str
    grid_color: str
    city_color: str
    tour_color: str
    best_tour_color: str
    
    # Chart colors
    chart_bg: str
    chart_line1: str
    chart_line2: str
    chart_line3: str
    chart_fill: str


DARK_THEME = Theme(
    name='dark',
    
    # Main colors - Cyberpunk-inspired
    background='#0a0a0f',
    surface='#141420',
    primary='#6366f1',      # Indigo
    secondary='#818cf8',    # Light indigo
    accent='#22d3ee',       # Cyan
    
    # Text colors
    text_primary='#f1f5f9',
    text_secondary='#94a3b8',
    text_muted='#475569',
    
    # Status colors
    success='#10b981',      # Emerald
    warning='#f59e0b',      # Amber
    error='#ef4444',        # Red
    
    # Canvas colors
    canvas_bg='#0f0f1a',
    grid_color='#1e1e2e',
    city_color='#22d3ee',
    tour_color='#6366f1',
    best_tour_color='#10b981',
    
    # Chart colors
    chart_bg='#141420',
    chart_line1='#6366f1',
    chart_line2='#22d3ee',
    chart_line3='#f59e0b',
    chart_fill='rgba(99, 102, 241, 0.2)',
)


LIGHT_THEME = Theme(
    name='light',
    
    # Main colors - Clean modern
    background='#f8fafc',
    surface='#ffffff',
    primary='#4f46e5',
    secondary='#6366f1',
    accent='#0891b2',
    
    # Text colors
    text_primary='#0f172a',
    text_secondary='#475569',
    text_muted='#94a3b8',
    
    # Status colors
    success='#059669',
    warning='#d97706',
    error='#dc2626',
    
    # Canvas colors
    canvas_bg='#ffffff',
    grid_color='#e2e8f0',
    city_color='#0891b2',
    tour_color='#4f46e5',
    best_tour_color='#059669',
    
    # Chart colors
    chart_bg='#ffffff',
    chart_line1='#4f46e5',
    chart_line2='#0891b2',
    chart_line3='#d97706',
    chart_fill='rgba(79, 70, 229, 0.1)',
)


class ThemeManager:
    """Manages application themes."""
    
    def __init__(self):
        self.themes = {
            'dark': DARK_THEME,
            'light': LIGHT_THEME,
        }
        self.current_theme = DARK_THEME
    
    def set_theme(self, name: str):
        """Set active theme by name."""
        if name in self.themes:
            self.current_theme = self.themes[name]
    
    def get_theme(self) -> Theme:
        """Get current theme."""
        return self.current_theme
    
    def get_stylesheet(self) -> str:
        """Generate Qt stylesheet for current theme."""
        t = self.current_theme
        return f"""
            QMainWindow {{
                background-color: {t.background};
            }}
            
            QWidget {{
                background-color: {t.background};
                color: {t.text_primary};
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif;
            }}
            
            QLabel {{
                color: {t.text_primary};
                font-size: 13px;
            }}
            
            QLabel[class="heading"] {{
                font-size: 16px;
                font-weight: 600;
                color: {t.text_primary};
            }}
            
            QLabel[class="subheading"] {{
                font-size: 12px;
                color: {t.text_secondary};
            }}
            
            QPushButton {{
                background-color: {t.primary};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                font-size: 13px;
            }}
            
            QPushButton:hover {{
                background-color: {t.secondary};
            }}
            
            QPushButton:pressed {{
                background-color: {t.primary};
            }}
            
            QPushButton:disabled {{
                background-color: {t.text_muted};
                color: {t.text_secondary};
            }}
            
            QPushButton[class="secondary"] {{
                background-color: {t.surface};
                color: {t.text_primary};
                border: 1px solid {t.text_muted};
            }}
            
            QPushButton[class="secondary"]:hover {{
                background-color: {t.text_muted};
            }}
            
            QPushButton[class="success"] {{
                background-color: {t.success};
            }}
            
            QPushButton[class="warning"] {{
                background-color: {t.warning};
            }}
            
            QPushButton[class="danger"] {{
                background-color: {t.error};
            }}
            
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {t.text_muted};
                border-radius: 3px;
            }}
            
            QSlider::handle:horizontal {{
                background: {t.primary};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {t.secondary};
            }}
            
            QSlider::sub-page:horizontal {{
                background: {t.primary};
                border-radius: 3px;
            }}
            
            QComboBox {{
                background-color: {t.surface};
                color: {t.text_primary};
                border: 1px solid {t.text_muted};
                border-radius: 6px;
                padding: 6px 12px;
                padding-right: 28px;
                min-width: 120px;
                min-height: 24px;
                font-size: 13px;
            }}
            
            QComboBox:hover {{
                border-color: {t.primary};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 28px;
                subcontrol-origin: padding;
                subcontrol-position: right center;
            }}
            
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 6px solid {t.text_secondary};
            }}
            
            QComboBox::down-arrow:hover {{
                border-top-color: {t.primary};
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {t.surface};
                color: {t.text_primary};
                selection-background-color: {t.primary};
                selection-color: white;
                border: 1px solid {t.text_muted};
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }}
            
            QComboBox QAbstractItemView::item {{
                min-height: 28px;
                padding: 4px 8px;
            }}
            
            QComboBox QAbstractItemView::item:hover {{
                background-color: {t.primary};
                color: white;
            }}
            
            QCheckBox {{
                color: {t.text_primary};
                font-size: 13px;
                spacing: 8px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {t.text_muted};
                border-radius: 4px;
                background-color: {t.surface};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {t.primary};
                border-color: {t.primary};
            }}
            
            QCheckBox::indicator:hover {{
                border-color: {t.primary};
            }}
            
            QSpinBox, QDoubleSpinBox {{
                background-color: {t.surface};
                color: {t.text_primary};
                border: 1px solid {t.text_muted};
                border-radius: 6px;
                padding: 6px 12px;
            }}
            
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {t.primary};
            }}
            
            QGroupBox {{
                background-color: {t.surface};
                border: 1px solid {t.text_muted};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: 500;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: {t.text_secondary};
            }}
            
            QProgressBar {{
                background-color: {t.surface};
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {t.primary};
                border-radius: 4px;
            }}
            
            QTabWidget::pane {{
                background-color: {t.surface};
                border: 1px solid {t.text_muted};
                border-radius: 8px;
            }}
            
            QTabBar::tab {{
                background-color: {t.background};
                color: {t.text_secondary};
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {t.surface};
                color: {t.text_primary};
            }}
            
            QTabBar::tab:hover {{
                color: {t.primary};
            }}
            
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            
            QScrollArea > QWidget > QWidget {{
                background-color: transparent;
            }}
            
            QScrollBar:vertical {{
                background: {t.background};
                width: 10px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical {{
                background: {t.text_muted};
                border-radius: 5px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background: {t.text_secondary};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar:horizontal {{
                background: {t.background};
                height: 10px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:horizontal {{
                background: {t.text_muted};
                border-radius: 5px;
                min-width: 20px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background: {t.text_secondary};
            }}
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            QMenuBar {{
                background-color: {t.surface};
                color: {t.text_primary};
                border-bottom: 1px solid {t.text_muted};
            }}
            
            QMenuBar::item {{
                padding: 6px 12px;
            }}
            
            QMenuBar::item:selected {{
                background-color: {t.primary};
                color: white;
            }}
            
            QMenu {{
                background-color: {t.surface};
                color: {t.text_primary};
                border: 1px solid {t.text_muted};
                border-radius: 6px;
            }}
            
            QMenu::item {{
                padding: 8px 24px;
            }}
            
            QMenu::item:selected {{
                background-color: {t.primary};
                color: white;
            }}
            
            QStatusBar {{
                background-color: {t.surface};
                color: {t.text_secondary};
                border-top: 1px solid {t.text_muted};
            }}
        """
    
    def get_matplotlib_style(self) -> Dict:
        """Get matplotlib style parameters for current theme."""
        t = self.current_theme
        return {
            'figure.facecolor': t.chart_bg,
            'axes.facecolor': t.chart_bg,
            'axes.edgecolor': t.text_muted,
            'axes.labelcolor': t.text_primary,
            'axes.titlecolor': t.text_primary,
            'xtick.color': t.text_secondary,
            'ytick.color': t.text_secondary,
            'text.color': t.text_primary,
            'grid.color': t.grid_color,
            'legend.facecolor': t.surface,
            'legend.edgecolor': t.text_muted,
        }
