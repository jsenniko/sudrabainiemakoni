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
        Canvas.__init__(self, self.fig)
        self.initplot()
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
        # Reset navigation toolbar state after axes recreation
        self.reset_navigation_state()
        
    def reset_navigation_state(self):
        """Reset navigation toolbar state after axes recreation"""
        if hasattr(self, '_mpl_widget_toolbar') and self._mpl_widget_toolbar is not None:
            toolbar = self._mpl_widget_toolbar
            try:
                # Clear navigation history stack
                if hasattr(toolbar, '_nav_stack'):
                    toolbar._nav_stack.clear()
                elif hasattr(toolbar, 'nav_stack'):
                    toolbar.nav_stack.clear()
                
                # Mark that we need to set home view on next draw
                self._need_home_view = True
                    
                # Ensure cursor is in normal state
                if hasattr(toolbar, '_lastCursor'):
                    toolbar._lastCursor = None
                    
                # Reset any active mode (zoom/pan)
                if hasattr(toolbar, '_active'):
                    toolbar._active = None
                if hasattr(toolbar, 'mode'):
                    toolbar.mode = ''
                    
            except Exception as e:
                # Silently handle any toolbar state reset errors
                pass
                
    def set_home_view(self):
        """Set current view as home view for navigation toolbar"""
        if hasattr(self, '_mpl_widget_toolbar') and self._mpl_widget_toolbar is not None:
            toolbar = self._mpl_widget_toolbar
            try:
                if hasattr(toolbar, 'push_current'):
                    toolbar.push_current()
            except Exception as e:
                pass
                
    def draw(self):
        """Override draw to automatically set home view after drawing"""
        # Call the parent draw method
        super().draw()
        # Only set home view if flagged by reset_navigation_state
        if hasattr(self, '_need_home_view') and self._need_home_view:
            self.set_home_view()
            self._need_home_view = False
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Store toolbar reference in canvas for reset functionality
        self.canvas._mpl_widget_toolbar = self.toolbar
        
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        
        
        
