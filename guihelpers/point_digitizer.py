import matplotlib
# Set backend for WSL compatibility
matplotlib.use('TkAgg')  # Use TkAgg backend for WSL
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, simpledialog

class MplPointDigitizer:
    def __init__(self, ax):        
        self.ax = ax
        # Object tracking
        # data used for connjection to external objects
        self._points = []  # [{'marker': Line2D, 'annotation': Annotation, 'data':Any}]
        
        # Selection and interaction state
        self._selected_point = None
        self._selection_highlight = None
        self.hit_radius = 15  # pixels for hit detection
        
        # Drag state
        self._dragging_point = None
        self._drag_start_pos = None
        
        # Event connections
        self._event_connections = {}

        self.is_digitizing = False
        
        # Callback system for external integration
        self.callbacks = {
            'on_point_added': None,      # Called when new point is added
            'on_point_selected': None,   # Called when point is selected
            'on_point_moved': None,      # Called when point is moved
            'on_point_removed': None,    # Called when point is removed
            'on_context_menu': None,     # Called on right-click
            'on_digitization_start': None, # Called when digitization starts
            'on_digitization_stop': None,  # Called when digitization stops
            'get_point_name': None,      # Called to get name for new point
            'get_point_data': None,      # Called to get data for new point
            'get_precise_position': None, # Called to get precise position (x, y, ctrl_pressed) -> (new_x, new_y) or None
        }
        
        # Default context menu settings
        self.show_default_context_menu = True

    def set_callback(self, event_name, callback_func):
        """Register a callback function for a specific event."""
        if event_name in self.callbacks:
            self.callbacks[event_name] = callback_func
        else:
            raise ValueError(f"Unknown callback event: {event_name}")
    def set_default_context_menu(self, enabled=True):
        """Enable or disable the default context menu."""
        self.show_default_context_menu = enabled
    
    def _get_callback(self, event_name):
        return self.callbacks.get(event_name)    
    def _call_callback(self, event_name, *args, **kwargs):
        """Call a registered callback if it exists."""
        callback = self._get_callback(event_name)
        if callback and callable(callback):
            return callback(*args, **kwargs)
        return None
    
    def _show_default_context_menu(self, point_index, event):
        """Show default context menu for a point."""
        if not self._index_valid(point_index):
            return
            
        point = self._points[point_index]
        
        # Create context menu for desktop backends
        try:
            menu = tk.Menu(self.ax.figure.canvas.get_tk_widget(), tearoff=0)
            
            menu.add_command(label=f"Change name of '{point['name']}'", 
                           command=lambda: self._change_point_name(point_index))
            menu.add_command(label=f"Change coordinates", 
                           command=lambda: self._change_point_coordinates(point_index))
            menu.add_separator()
            menu.add_command(label=f"Delete '{point['name']}'", 
                           command=lambda: self._delete_point_with_confirmation(point_index))
            
            # Show menu at mouse position
            if hasattr(event, 'guiEvent'):
                menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            else:
                # Fallback for events without guiEvent
                menu.tk_popup(event.x + 50, event.y + 50)
                
        except Exception as e:
            print(f"Error showing context menu: {e}")
    
    def _change_point_name(self, point_index):
        """Change the name of a point."""
        if not self._index_valid(point_index):
            return
            
        point = self._points[point_index]
        current_name = point['name']
        
        new_name = simpledialog.askstring(
            "Change Point Name", 
            f"Enter new name for point:",
            initialvalue=current_name
        )
        
        if new_name and new_name != current_name:
            point['name'] = new_name
            self.update_name(point_index, new_name)
    
    def _change_point_coordinates(self, point_index):
        """Change the coordinates of a point."""
        if not self._index_valid(point_index):
            return
            
        point = self._points[point_index]
        current_x, current_y = point['x'], point['y']
        
        # Get new X coordinate
        new_x = simpledialog.askfloat(
            "Change X Coordinate", 
            f"Enter new X coordinate for '{point['name']}':",
            initialvalue=current_x
        )
        
        if new_x is None:
            return
            
        # Get new Y coordinate
        new_y = simpledialog.askfloat(
            "Change Y Coordinate", 
            f"Enter new Y coordinate for '{point['name']}':",
            initialvalue=current_y
        )
        
        if new_y is None:
            return
            
        # Update point position
        self.update_point_position(point_index, new_x, new_y)
    
    def _delete_point_with_confirmation(self, point_index):
        """Delete a point with confirmation dialog."""
        if not self._index_valid(point_index):
            return
            
        point = self._points[point_index]
        
        result = messagebox.askyesno(
            "Confirm Delete", 
            f"Are you sure you want to delete point '{point['name']}'?"
        )
        
        if result:
            self.remove_point(point_index)

    def setup_events(self):
        """Connect to matplotlib events for interaction."""
        canvas = self.ax.figure.canvas
        self._event_connections['press'] = canvas.mpl_connect(
            'button_press_event', self._on_mouse_press)
        self._event_connections['motion'] = canvas.mpl_connect(
            'motion_notify_event', self._on_mouse_motion)
        self._event_connections['release'] = canvas.mpl_connect(
            'button_release_event', self._on_mouse_release)
    
    def disconnect_events(self):
        """Disconnect from matplotlib events."""
        canvas = self.ax.figure.canvas
        for connection in self._event_connections.values():
            canvas.mpl_disconnect(connection)
        self._event_connections.clear()

    def _draw_annotation(self, ax, x, y, name):
        annotation = ax.annotate(
            name,
            xy=(x, y),
            xytext=(3, 3),
            color='#AAFFAA',
            fontsize=16,
            textcoords='offset pixels'
        )
        return annotation
    def _draw_point(self, ax, x, y):
        marker = ax.plot(x, y, marker='o', fillstyle='none')[0]
        return marker

    def add_point(self, x, y, name, data = None):
        # Create marker (point)
        marker = self._draw_point(self.ax, x, y)
        # Create annotation (text label)
        annotation = self._draw_annotation(self.ax, x,y,name)
        
        new_point = {
            'x':x,
            'y':y,
            'name': name,
            'marker': marker,
            'annotation': annotation,
            'data': data,
        }
        # Store both objects
        self._points.append(new_point)
        
        # Efficient redraw that preserves zoom
        self.ax.figure.canvas.draw()
        
        point_index = len(self._points) - 1
        # Call external callback
        self._call_callback('on_point_added', point_index, new_point)
        
        return point_index
    def _index_valid(self, point_index):
        return (point_index>=0) and (point_index<len(self._points))
    
    def update_point_position(self, point_index, new_x, new_y):
        if not self._index_valid(point_index):
            return
            
        object = self._points[point_index]
        old_x, old_y = object['x'], object['y']
        
        # Update stored coordinates
        object['x'] = new_x
        object['y'] = new_y
        
        # Update marker position
        object['marker'].set_data([new_x], [new_y])
        
        # Update annotation position
        object['annotation'].xy = (new_x, new_y)
        
        # Update selection highlight if this star is selected
        if self._selected_point == point_index and self._selection_highlight:
            self._selection_highlight.set_data([new_x], [new_y])
        
        # Efficient redraw that preserves zoom
        self.ax.figure.canvas.draw()
        
        # Call external callback
        self._call_callback('on_point_moved', point_index, object, old_x, old_y, new_x, new_y)
    def update_name(self, point_index, new_name):
        if not self._index_valid(point_index):
            return
            
        object = self._points[point_index]
        object['annotation'].set_text(new_name)
        
        # Efficient redraw that preserves zoom
        self.ax.figure.canvas.draw()
    
    def remove_point(self, point_index):
        if not self._index_valid(point_index):
            return
            
        object = self._points[point_index]
        
        # Remove from axes
        object['marker'].remove()
        object['annotation'].remove()
        
        # Clear selection if this star was selected
        if self._selected_point == point_index:
            self.clear_selection()
        
        # Call external callback before removal
        self._call_callback('on_point_removed', point_index, object)
        
        # Remove from tracking
        #del self._points[point_index]
        self._points.remove(self._points[point_index])
        
        # Efficient redraw that preserves zoom
        self.ax.figure.canvas.draw()
    
    def clear_all_points(self):
        """Remove all star visuals from the plot."""
        self.clear_selection()
        for point_index in range(len(self._points)-1,-1,-1):
            object = self._points[point_index]
            object['marker'].remove()
            object['annotation'].remove()
        self._points = []
        self.ax.figure.canvas.draw()
    
    def show_selection(self, point_index):
        if not self._index_valid(point_index):
            return
        self.clear_selection()
            
        self._selected_point = point_index
        point = self._points[point_index]
        x, y = point['x'], point['y']
        
        # Create selection highlight (red square)
        self._selection_highlight = self.ax.plot(
            x, y,
            marker='s',  # square marker
            markersize=12,
            fillstyle='none',
            markeredgecolor='red',
            markeredgewidth=2,
            markerfacecolor='none'
        )[0]
        
        self.ax.figure.canvas.draw()
        
        # Call external callback
        self._call_callback('on_point_selected', point_index, point)
    
    def clear_selection(self):
        """Clear the current selection highlight."""
        if self._selection_highlight:
            try:
                self._selection_highlight.remove()
                self.ax.figure.canvas.draw()
            except ValueError:
                # Already removed
                pass
        
        self._selected_point = None
        self._selection_highlight = None
    def _find_point_at_position(self, event):
        """
        Find point at mouse position by checking both marker and annotation.
        
        Args:
            event: matplotlib mouse event
            
        """
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return None
            
        min_distance = float('inf')
        closest_point = None
        for point_index, object in enumerate(self._points):
            # Check marker hit
            marker_distance = self._get_marker_distance(object['marker'], event)
            #print(f'Find {point_index} {marker_distance}')
            if marker_distance < self.hit_radius and marker_distance < min_distance:
                min_distance = marker_distance
                closest_point = point_index
                
            # Check annotation hit (larger tolerance)
            annotation_distance = self._get_annotation_distance(object['annotation'], event)
            if annotation_distance < self.hit_radius * 1.5 and annotation_distance < min_distance:
                min_distance = annotation_distance
                closest_point = point_index
        
        return closest_point
    
    def _get_marker_distance(self, marker, event):
        # in general marker can be at different location that point['x'],point['y']
        """Calculate distance in screen coordinates from event to marker."""
        marker_data = marker.get_data()
        if len(marker_data[0]) == 0:
            return float('inf')
            
        marker_x, marker_y = marker_data[0][0], marker_data[1][0]
        marker_disp = self.ax.transData.transform((marker_x, marker_y))
        #print(f'{marker_x=} {marker_y=} {marker_disp=} {event.x=} {event.y=}')

        distance = ((event.x - marker_disp[0])**2 + (event.y - marker_disp[1])**2)**0.5
        return distance
    
    def _get_annotation_distance(self, annotation, event):
        # in general annotation.xy can be at different location that point['x'],point['y']
        """Calculate distance in screen coordinates from event to annotation."""
        ann_x, ann_y = annotation.xy
        ann_disp = self.ax.transData.transform((ann_x, ann_y))
        distance = ((event.x - ann_disp[0])**2 + (event.y - ann_disp[1])**2)**0.5
        return distance
    
    def _is_zooming_or_panning(self):
        """
        Robust detection of whether zooming or panning is active.
        """
        navigation_mode = self.ax.get_navigate_mode()
        is_zooming = navigation_mode is not None
        if not is_zooming:
            try: 
                is_zooming = fig.canvas.toolbar.mode!=''
            except:
                pass
        return is_zooming
    
    def _on_mouse_press(self, event):
        """Handle mouse press events."""
        # Get axes from event
        ax = event.inaxes
        if ax is None:
            return
            
        # Check if we're zooming or panning - more robust detection
        zooming_panning = self._is_zooming_or_panning()
        if zooming_panning:
            return
        
        if event.button == 1:  # Left click
            self._handle_left_click(event.xdata, event.ydata, event)
        elif event.button == 3:  
            self._handle_right_click(event.xdata, event.ydata, event)
    
    def _get_name_and_data(self, event):
        name =  None
        data = self._call_callback('get_point_data', event.xdata, event.ydata)            
        if data is not None:
            try:
                if isinstance(data, tuple) and len(data) == 2:
                    data, name = data
                else:
                    name = data['name']
            except:
                try:
                    name = data.name
                except:
                    name = None
        if name is None:
            name = self._call_callback('get_point_name', event.xdata, event.ydata)
        if name is None:
            name = 'New Point'  # Default name
        return name, data
    
    def _handle_left_click(self, x, y, event):
        #print(f'left click {event}')
        # Check for Ctrl key modifier in mouse events
        ctrl_pressed = hasattr(event, 'modifiers') and 'ctrl' in event.modifiers

        point_index = self._find_point_at_position(event)
        if point_index is not None:
            # Click on existing star
            if self._selected_point == point_index:
                # Second click on selected star - start dragging
                self._start_drag(point_index, event)
            else:
                # First click - select star
                self.show_selection(point_index)
        else:
            # Click on empty space - could add new star
            self.clear_selection()
            # Get name and data from external callbacks
            
            name, data = self._get_name_and_data(event)
            valid = data is not None or self._get_callback('get_point_data') is None
            if valid:
                # Get precise position if callback is available
                precise_pos = self._call_callback('get_precise_position', event.xdata, event.ydata, ctrl_pressed)
                if precise_pos is not None:
                    final_x, final_y = precise_pos
                else:
                    final_x, final_y = event.xdata, event.ydata
                    
                point_index = self.add_point(final_x, final_y, name, data = data)
                if point_index is not None:
                    # auto-select after adding                
                    self.show_selection(point_index)
    def _handle_right_click(self, _x, _y, event):
        point_index = self._find_point_at_position(event)
        if point_index is not None:
            self.show_selection(point_index)
            
            # Call external context menu callback first
            external_handled = self._call_callback('on_context_menu', point_index, self._points[point_index], event)
            
            # If no external callback or it returns False, show default menu
            if self.show_default_context_menu and (external_handled is None or external_handled is False):
                self._show_default_context_menu(point_index, event)
        else:
            self.stop_digitization()
    def _mark_digitization(self):
        self.ax.figure.set_facecolor('mistyrose')
        self.ax.figure.canvas.draw()

    def start_digitization(self):
        if not self.is_digitizing:
            self._mark_digitization()
            self.setup_events()
            self.is_digitizing = True
            # Call external callback
            self._call_callback('on_digitization_start', self._points)

    def stop_digitization(self):
        self.is_digitizing = False
        self.disconnect_events()
        self.clear_selection()
        self.ax.figure.set_facecolor('white')
        # Call external callback
        self._call_callback('on_digitization_stop', self._points)
        self.clear_all_points()

    def _on_mouse_motion(self, event):
        """Handle mouse motion during drag."""
        if self._dragging_point is not None:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                self.update_point_position(self._dragging_point, event.xdata, event.ydata)
    
    def _on_mouse_release(self, event):
        """Handle mouse release to end drag."""
        if self._dragging_point is not None:
            # Check for Ctrl key modifier in mouse events
            ctrl_pressed = hasattr(event, 'modifiers') and 'ctrl' in event.modifiers
            
            # Get precise position if callback is available
            point = self._points[self._dragging_point]
            precise_pos = self._call_callback('get_precise_position', point['x'], point['y'], ctrl_pressed)
            if precise_pos is not None:
                final_x, final_y = precise_pos
                self.update_point_position(self._dragging_point, final_x, final_y)
            
            self._dragging_point = None
            self._drag_start_pos = None
    
    def _start_drag(self, point_index, event):
        """Start dragging a star."""
        self._dragging_point = point_index
        self._drag_start_pos = (event.xdata, event.ydata)
        

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot([0,1],[0,1])
    ax.grid()

    pd = MplPointDigitizer(ax)
    pd.start_digitization()
    plt.show()