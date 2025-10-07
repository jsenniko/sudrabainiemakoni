"""
ControlPointDigitizer - Enhanced control point digitization using MplPointDigitizer.
Wrapper class that provides control point-specific functionality while leveraging
the generic point digitization capabilities of MplPointDigitizer.

Handles paired point digitization between two images with real-time epiline display
and height calculation.
"""

import numpy as np

# Support both direct script execution and package import
try:
    from .point_digitizer import MplPointDigitizer
except ImportError:
    from point_digitizer import MplPointDigitizer


class ControlPointDigitizer:
    """
    Control point digitization manager that wraps two MplPointDigitizer instances
    for sequential paired point digitization between two images.
    
    Features:
    - Alternates between two images for point pair creation
    - Real-time epiline display when points are added
    - Paired point deletion (removes both points of a correspondence)
    - Live height calculation and display
    - Point movement with epiline updates
    """
    
    def __init__(self, ax1, ax2, cloudimage_pair, main_window=None, z1=75, z2=90):
        """
        Initialize the control point digitizer.
        
        Args:
            ax1: matplotlib axes object for first image
            ax2: matplotlib axes object for second image
            cloudimage_pair: CloudImagePair object containing correspondence data
            main_window: main window reference for compatibility
            z1, z2: height range in km for epiline calculation (default: 75-90 km)
        """
        self.ax1 = ax1
        self.ax2 = ax2
        self.cloudimage_pair = cloudimage_pair
        self.main_window = main_window
        self.z1, self.z2 = z1, z2
        
        # Create two point digitizers
        self.digitizer1 = MplPointDigitizer(ax1)
        self.digitizer2 = MplPointDigitizer(ax2)
        
        # Disable default context menus, use control point-specific ones
        self.digitizer1.set_default_context_menu(False)
        self.digitizer2.set_default_context_menu(False)
        
        # State management
        self.current_digitizer = 0  # 0 for image1, 1 for image2
        self.temp_epilines = []  # Store temporary epiline plots
        
        # Store original spine properties for restoration
        self.original_spine_props = self._store_original_spine_properties()
        
        # Setup callbacks for both digitizers
        self._setup_callbacks()
        
        # Track if digitization is active
        self.is_digitizing = False
    
    def _store_original_spine_properties(self):
        """Store original spine properties for both axes."""
        props = {
            'ax1': {},
            'ax2': {}
        }
        
        # Store ax1 spine properties
        for spine_name, spine in self.ax1.spines.items():
            props['ax1'][spine_name] = {
                'color': spine.get_edgecolor(),
                'linewidth': spine.get_linewidth(),
                'visible': spine.get_visible()
            }
        
        # Store ax2 spine properties  
        for spine_name, spine in self.ax2.spines.items():
            props['ax2'][spine_name] = {
                'color': spine.get_edgecolor(),
                'linewidth': spine.get_linewidth(),
                'visible': spine.get_visible()
            }
        
        return props
        
    def _setup_callbacks(self):
        """Setup all control point-specific callbacks for both digitizers."""
        # Digitizer 1 callbacks
        self.digitizer1.set_callback('get_point_data', 
                                    lambda x, y: self._create_point_data(x, y, 0))
        self.digitizer1.set_callback('on_point_added', 
                                    lambda idx, point: self._on_point_added(0, idx, point))
        self.digitizer1.set_callback('on_point_moved', 
                                    lambda idx, point, old_x, old_y, new_x, new_y: 
                                    self._on_point_moved(0, idx, point, old_x, old_y, new_x, new_y))
        self.digitizer1.set_callback('on_point_removed', 
                                    lambda idx, point: self._on_point_removed(0, idx, point))
        self.digitizer1.set_callback('on_context_menu', 
                                    lambda idx, point, event: self._show_control_point_context_menu(0, idx, point, event))
        self.digitizer1.set_callback('on_point_selected', 
                                    lambda idx, point: self._on_point_selected(0, idx, point))
        self.digitizer1.set_callback('on_digitization_stop', self._save_control_points)
        
        # Digitizer 2 callbacks
        self.digitizer2.set_callback('get_point_data', 
                                    lambda x, y: self._create_point_data(x, y, 1))
        self.digitizer2.set_callback('on_point_added', 
                                    lambda idx, point: self._on_point_added(1, idx, point))
        self.digitizer2.set_callback('on_point_moved', 
                                    lambda idx, point, old_x, old_y, new_x, new_y: 
                                    self._on_point_moved(1, idx, point, old_x, old_y, new_x, new_y))
        self.digitizer2.set_callback('on_point_removed', 
                                    lambda idx, point: self._on_point_removed(1, idx, point))
        self.digitizer2.set_callback('on_context_menu', 
                                    lambda idx, point, event: self._show_control_point_context_menu(1, idx, point, event))
        self.digitizer2.set_callback('on_point_selected', 
                                    lambda idx, point: self._on_point_selected(1, idx, point))
        self.digitizer2.set_callback('on_digitization_stop', self._save_control_points)
    
    def start_digitization(self):
        """Start control point digitization mode."""
        
        if not self.is_digitizing:
            self._load_existing_control_points()
            
            # Start both digitizers
            self.digitizer1.start_digitization()
            self.digitizer2.start_digitization()
            
            # Mark the first digitizer as active
            self._highlight_active_digitizer()
            
            self.is_digitizing = True
    
    def stop_digitization(self):
        """Stop control point digitization mode."""
        if self.is_digitizing:
            self.digitizer1.stop_digitization()
            self.digitizer2.stop_digitization()
            
            # Clear epilines
            self._clear_epilines()
            
            # Reset borders to default
            self._reset_borders()
            
            self.is_digitizing = False
    
    def set_height_range(self, z1, z2):
        """Set the height range for epiline calculation."""
        self.z1, self.z2 = z1, z2
    
    def _create_point_data(self, x, y, digitizer_id):
        """Create data structure for a control point."""
        
        if self.current_digitizer != digitizer_id:
            return None  # Only allow addition to active digitizer
        
        # Get the next correspondence index
        correspondence_idx = len(self.cloudimage_pair.correspondances[digitizer_id])
        
        # Create point name and include in data
        point_name = f"{correspondence_idx + 1}"
        
        return {
            'correspondence_index': correspondence_idx,
            'digitizer_id': digitizer_id,
            'x': x,
            'y': y,
            'paired': False,
            'name': point_name  # Include name in data so _get_name_and_data can extract it
        }
    
    def _on_point_added(self, digitizer_id, point_index, point):
        """Handle when a control point is added interactively by user."""
        
        if point['data'] is None:
            return  # Point was rejected
        
        x, y = point['x'], point['y']
        correspondence_idx = point['data']['correspondence_index']
        
        
        # Add to correspondances array (only called for interactive additions)
        self.cloudimage_pair.correspondances[digitizer_id] = np.append(
            self.cloudimage_pair.correspondances[digitizer_id], [[x, y]], axis=0)
        
        # Point name was already set in _create_point_data
        point_name = point['data']['name']
        
        # Show epilines and calculate height (without redraw for epilines)
        self._show_point_correspondence(digitizer_id, x, y, correspondence_idx, redraw=False)
        
        # Switch to other digitizer for next point (without redraw)
        old_digitizer = self.current_digitizer
        self.current_digitizer = (self.current_digitizer + 1) % 2
        self._highlight_active_digitizer(redraw=False)
        
        # Single final redraw for epilines and highlighting 
        # (MplPointDigitizer already did one redraw, but we need one more for epilines/borders)
        self.ax1.figure.canvas.draw()
        
        # Mark points as paired if correspondence is complete
        if self._correspondence_pair_complete(correspondence_idx):
            point['data']['paired'] = True
            # Also mark the paired point
            paired_digitizer = self._get_digitizer(1 - digitizer_id)
            if correspondence_idx < len(paired_digitizer._points):
                paired_digitizer._points[correspondence_idx]['data']['paired'] = True
    
    def _on_point_moved(self, digitizer_id, point_index, point, old_x, old_y, new_x, new_y):
        """Handle when a control point is moved."""
        correspondence_idx = point['data']['correspondence_index']
        
        # Update the correspondances array
        self.cloudimage_pair.correspondances[digitizer_id][correspondence_idx] = [new_x, new_y]
        
        # Clear old epilines
        self._clear_epilines()
        
        # Show updated epilines and calculate height (no redraw needed)
        # MplPointDigitizer.update_point_position() already calls canvas.draw() to update point position
        # Since both axes share the same figure, this redraw also displays our epiline changes
        self._show_point_correspondence(digitizer_id, new_x, new_y, correspondence_idx, redraw=False)
        
    
    def _on_point_removed(self, digitizer_id, _point_index, point):
        """Handle when a single control point is removed - remove from correspondances and reindex."""
        correspondence_idx = point['data']['correspondence_index']
        
        # Check if correspondence still exists (might have been deleted already)
        if correspondence_idx >= len(self.cloudimage_pair.correspondances[digitizer_id]):
            return
        
        print(f"Removing point from digitizer {digitizer_id}, correspondence {correspondence_idx + 1}")
        
        # Remove this specific point from its correspondances array
        self.cloudimage_pair.correspondances[digitizer_id] = np.delete(
            self.cloudimage_pair.correspondances[digitizer_id], correspondence_idx, axis=0)
        
        # Clear epilines
        self._clear_epilines()
        
    
    def _on_point_selected(self, digitizer_id, point_index, point):
        """Handle when a control point is selected - show epilines."""
        if point['data'] is None:
            return
            
        x, y = point['x'], point['y']
        correspondence_idx = point['data']['correspondence_index']
        point_name = point['data']['name']
        
        
        # Show epilines and calculate height for the selected point
        self._show_point_correspondence(digitizer_id, x, y, correspondence_idx, redraw=True)
    
    def _show_point_correspondence(self, digitizer_id, x, y, correspondence_idx, redraw=True):
        """Show epilines and calculate height for a control point correspondence."""
        # Show epilines on the opposite image
        if digitizer_id == 0:
            self._show_epilines_for_point(x, y, from_image=0, to_image=1, redraw=redraw)
            if len(self.cloudimage_pair.correspondances[1]) == len(self.cloudimage_pair.correspondances[0]):
                self._show_epilines_for_point(
                    self.cloudimage_pair.correspondances[1][correspondence_idx][0],
                    self.cloudimage_pair.correspondances[1][correspondence_idx][1],
                    from_image=1, to_image=0, redraw=redraw, clear_previous=False)
        else: 
            self._show_epilines_for_point(x, y, from_image=1, to_image=0, redraw=redraw)
            if len(self.cloudimage_pair.correspondances[1]) == len(self.cloudimage_pair.correspondances[0]):
                self._show_epilines_for_point(
                    self.cloudimage_pair.correspondances[0][correspondence_idx][0],
                    self.cloudimage_pair.correspondances[0][correspondence_idx][1],
                    from_image=0, to_image=1, redraw=redraw, clear_previous=False)
        
        # If this point has a paired point, calculate and display height
        if self._correspondence_pair_complete(correspondence_idx):
            self._calculate_and_display_height(correspondence_idx)
    
    def _reindex_correspondence_points(self):
        """Update correspondence indices after deletion without triggering redraws."""
        for digitizer in [self.digitizer1, self.digitizer2]:
            for point_idx, point in enumerate(digitizer._points):
                # Update correspondence index to match array position
                point['data']['correspondence_index'] = point_idx
                # Update point name using update_name with redraw=False
                new_name = f"{point_idx + 1}"
                point['name'] = new_name
                digitizer.update_name(point_idx, new_name, redraw=False)
        
        # Single redraw after all updates are complete
        self.ax1.figure.canvas.draw()
    
    def _show_control_point_context_menu(self, digitizer_id, point_index, point, event):
        """Show PyQt5 context menu for control point operations."""
        correspondence_idx = point['data']['correspondence_index']
        
        try:
            from PyQt5.QtWidgets import QMenu, QAction
            from PyQt5.QtCore import QPoint
            
            # Create context menu
            menu = QMenu()
            
            # Add menu actions
            edit_action = QAction(f"Edit coordinates of pair {correspondence_idx + 1}", menu)
            edit_action.triggered.connect(lambda: self._edit_correspondence_coordinates(correspondence_idx))
            menu.addAction(edit_action)
            
            delete_action = QAction(f"Delete correspondence pair {correspondence_idx + 1}", menu)
            delete_action.triggered.connect(lambda: self._delete_correspondence_pair(correspondence_idx))
            menu.addAction(delete_action)
                    
            
            # Convert matplotlib event coordinates to Qt coordinates
            ax = self._get_digitizer(digitizer_id).ax
            canvas_widget = ax.figure.canvas
            if hasattr(canvas_widget, 'mapToGlobal'):
                canvas_pos = QPoint(int(event.x), int(canvas_widget.height() - event.y))
                global_pos = canvas_widget.mapToGlobal(canvas_pos)
                menu.exec_(global_pos)
            else:
                menu.exec_()
                
        except ImportError:
            print("PyQt5 not available for context menu")
        except Exception as e:
            print(f"Error showing context menu: {e}")
        
        return True  # Handled externally
    
    def _show_epilines_for_point(self, x, y, from_image, to_image, redraw=True, clear_previous=True):
        """Show epilines on the opposite image for a point."""
        try:
            if from_image == 0:
                epilines = self.cloudimage_pair.GetEpilinesAtHeightInterval(
                    [self.z1, self.z2], [[x, y]], True)
                target_ax = self.ax2
            else:
                epilines = self.cloudimage_pair.GetEpilinesAtHeightInterval(
                    [self.z1, self.z2], [[x, y]], False)
                target_ax = self.ax1
            
            # Clear previous epilines
            if clear_previous:
                self._clear_epilines()
            
            # Plot new epilines
            epiline_plot = target_ax.plot(epilines[0, :, 0], epilines[0, :, 1],
                                        color='yellow', marker='o', ms=1, lw=0.8)
            self.temp_epilines.extend(epiline_plot)
            
            # Conditional redraw
            if redraw:
                target_ax.figure.canvas.draw()
            
        except Exception as e:
            print(f"Error showing epilines: {e}")
    
    def _clear_epilines(self):
        """Clear all temporary epilines."""
        for line in self.temp_epilines:
            try:
                line.remove()
            except ValueError:
                pass  # Already removed
        self.temp_epilines.clear()
    
    def _calculate_and_display_height(self, correspondence_idx):
        """Calculate and display height for a correspondence pair."""
        try:
            if correspondence_idx < len(self.cloudimage_pair.correspondances[0]) and \
               correspondence_idx < len(self.cloudimage_pair.correspondances[1]):
                
                pt1 = [self.cloudimage_pair.correspondances[0][correspondence_idx]]
                pt2 = [self.cloudimage_pair.correspondances[1][correspondence_idx]]
                
                llh, rayminimaldistance, _z_intrinsic_error, valid = \
                    self.cloudimage_pair.GetHeightPoints(pt1, pt2)
                
                height_km = llh[2][0] / 1000.0
                ray_distance = rayminimaldistance[0]
                print(f'Pair {correspondence_idx + 1}: Height {height_km:.1f}km, Ray distance {ray_distance:.1f}m')
                    
        except Exception as e:
            print(f"Error calculating height for pair {correspondence_idx + 1}: {e}")
    
    def _correspondence_pair_complete(self, correspondence_idx):
        """Check if both points of a correspondence pair exist."""
        return (correspondence_idx < len(self.cloudimage_pair.correspondances[0]) and 
                correspondence_idx < len(self.cloudimage_pair.correspondances[1]))
    
    def _get_digitizer(self, digitizer_id):
        """Get digitizer by ID."""
        return self.digitizer1 if digitizer_id == 0 else self.digitizer2
    
    def _highlight_active_digitizer(self, redraw=True):
        """Highlight the currently active digitizer with thick colored border."""
        
        # Colors for active/inactive borders
        active_color = 'red'       # Active digitizer - bright red border
        inactive_color = 'gray'    # Inactive digitizer - gray border
        active_width = 3           # Thick border for active
        inactive_width = 1         # Thin border for inactive
        
        # Set border for ax1 (digitizer 0)
        is_ax1_active = (self.current_digitizer == 0)
        color1 = active_color if is_ax1_active else inactive_color
        width1 = active_width if is_ax1_active else inactive_width
        
        for spine in self.ax1.spines.values():
            spine.set_color(color1)
            spine.set_linewidth(width1)
            spine.set_visible(True)
        
        # Set border for ax2 (digitizer 1) 
        is_ax2_active = (self.current_digitizer == 1)
        color2 = active_color if is_ax2_active else inactive_color
        width2 = active_width if is_ax2_active else inactive_width
        
        for spine in self.ax2.spines.values():
            spine.set_color(color2)
            spine.set_linewidth(width2)
            spine.set_visible(True)
        
        
        # Conditional redraw to show border changes
        if redraw:
            self.ax1.figure.canvas.draw()
        
    
    def _reset_borders(self):
        """Reset axes borders to original appearance."""

        for ax, axname in [(self.ax1, 'ax1'), (self.ax2, 'ax2')]:
            for spine_name, spine in ax.spines.items():
                original = self.original_spine_props[axname][spine_name]
                spine.set_color(original['color'])
                spine.set_linewidth(original['linewidth'])
                spine.set_visible(original['visible'])                        
        
        # Redraw
        self.ax1.figure.canvas.draw()
    
    def _load_existing_control_points(self):
        """Load existing control points from cloudimage_pair into the digitizers."""
        # Clear existing points
        self.digitizer1.clear_all_points()
        self.digitizer2.clear_all_points()
        
        # Add points from correspondances arrays (non-interactive = loading from data)
        for idx, (pt1, pt2) in enumerate(zip(self.cloudimage_pair.correspondances[0], 
                                           self.cloudimage_pair.correspondances[1])):
            # Create point names
            name1 = f"{idx + 1}"
            name2 = f"{idx + 1}"
            
            # Add point to digitizer 1 (include name in data for consistency)
            data1 = {
                'correspondence_index': idx,
                'digitizer_id': 0,
                'x': pt1[0],
                'y': pt1[1],
                'paired': True,
                'name': name1  # Include name in data
            }
            self.digitizer1.add_point(pt1[0], pt1[1], name1, data=data1, interactive=False)
            
            # Add point to digitizer 2 (include name in data for consistency)
            data2 = {
                'correspondence_index': idx,
                'digitizer_id': 1,
                'x': pt2[0],
                'y': pt2[1],
                'paired': True,
                'name': name2  # Include name in data
            }
            self.digitizer2.add_point(pt2[0], pt2[1], name2, data=data2, interactive=False)
        
        # Single redraw after loading all points
        self.digitizer1.batch_redraw()
    
    def _save_control_points(self, _points):
        """Save control points to file when digitization stops."""
        if self.main_window:
            self.main_window.isDigitizeControlPoints = None
        
        if self.cloudimage_pair is not None:
            from qthelper import gui_save_fname
            import os
            
            # Create default filename
            if hasattr(self.cloudimage_pair, 'cloudImage1') and hasattr(self.cloudimage_pair, 'cloudImage2'):
                filename = f'{os.path.split(self.cloudimage_pair.cloudImage1.filename)[0]}/' \
                          f'{self.cloudimage_pair.cloudImage1.code}_{self.cloudimage_pair.cloudImage2.code}.txt'
            else:
                filename = 'control_points.txt'
            
            filename = gui_save_fname(
                directory=filename,
                caption='Atbilstību fails',
                filter='(*.txt)')
            
            if filename != '':
                self.cloudimage_pair.SaveCorrespondances(filename)
                num_pairs = min(len(self.cloudimage_pair.correspondances[0]), 
                              len(self.cloudimage_pair.correspondances[1]))
                print(f"Saved {num_pairs} control point pairs to {filename}")
        
        if self.main_window:
            self.main_window.DrawImage(otrs=True, kontrolpunkti=True)
    
    def _edit_correspondence_coordinates(self, correspondence_idx):
        """Edit coordinates of both points in a correspondence pair."""
        # Implementation would show dialogs to edit both points
        print(f"Edit coordinates for correspondence pair {correspondence_idx + 1}")
    
    def _delete_correspondence_pair(self, correspondence_idx):
        """Delete an entire correspondence pair with confirmation."""
        from qthelper import gui_confirm
        
        confirm = gui_confirm(
            caption=f"Delete correspondence pair {correspondence_idx + 1}?"
        )
        
        if confirm:
            # Remove both points independently - each will trigger _on_point_removed
            points_to_remove = []
            
            # Find both points of the correspondence pair
            for digitizer_id, digitizer in enumerate([self.digitizer1, self.digitizer2]):
                for idx, point in enumerate(digitizer._points):
                    if point['data']['correspondence_index'] == correspondence_idx:
                        points_to_remove.append((digitizer, idx))
                        break
            
            # Remove both points (each triggers independent _on_point_removed)
            for digitizer, idx in points_to_remove:
                digitizer.remove_point(idx)
            self._reindex_correspondence_points()
    