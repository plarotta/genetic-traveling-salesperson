import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Path Visualizer")
        self.setGeometry(100, 100, 800, 600)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.paths = []  # List to store generated paths

    def plot_member(self, member, generation, member_fitness):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(member[:, 0], member[:, 1], marker='o', linestyle='-')
        ax.set_xlabel('X')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_ylabel('Y')
        ax.set_title(f'Best, gen: {generation}, fitness: {member_fitness:.2f}')
        ax.grid(True)

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
