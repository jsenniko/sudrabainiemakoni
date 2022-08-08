from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib
import matplotlib.pyplot as plt

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure(tight_layout=False)
        #self.ax = self.fig.add_subplot(111)
        self.initplot()
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
    def delaxes(self):
        for ax in self.fig.axes:
            self.fig.delaxes(ax)
        
    def initplot(self, axes=[111]):
        self.delaxes()
        axlist=[]
        for axpos in axes:
            ax=self.fig.add_subplot(axpos)
            axlist.append(ax)
        if len(axlist)==1:
            self.ax=axlist[0]
        else:
            self.ax=axlist
        self.fig.subplots_adjust(0.05,0.05,0.95,0.95)
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        
        
        
