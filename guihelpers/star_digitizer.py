"""
StarDigitizer - Enhanced star digitization using MplPointDigitizer.
Wrapper class that provides star-specific functionality while leveraging
the generic point digitization capabilities of MplPointDigitizer.
"""

import numpy as np
from sudrabainiemakoni.starreference import StarReference
from point_digitizer import MplPointDigitizer


def gui_star_input(caption='Ievadi zvaigznes informāciju'):
    """
    Enhanced star input dialog that supports multiple coordinate formats.
    Parsing is now handled by StarReference constructor.
    
    Returns:
        str: The input string or None if cancelled
    """
    from qthelper import gui_string
    
    full_caption = f"""{caption}
Formāti:
• Zvaigznes vārds: Sirius, Polaris
• Alt-Az (grādi): 120.5,45.2  (azimuts,augstums)
• RA/DEC (grādi): ra:83.633,22.014
Ievadi:"""
    
    return gui_string(caption=full_caption)


class StarDigitizer:
    """
    Star digitization manager that wraps MplPointDigitizer with star-specific functionality.
    Provides enhanced star input, coordinate validation, and PyQt5 context menus.
    """
    
    def __init__(self, ax, cloudimage, main_window=None, centroid_box_size=50):
        """
        Initialize the star digitizer.
        
        Args:
            ax: matplotlib axes object
            cloudimage: CloudImage object containing star references
            main_window: main window reference for compatibility
            centroid_box_size: size of the box around original position for centroiding (default: 10 pixels)
        """
        self.cloudimage = cloudimage
        self.main_window = main_window
        self.centroid_box_size = centroid_box_size
        
        # Create the underlying point digitizer
        self.point_digitizer = MplPointDigitizer(ax)
        
        # Disable default context menu, use star-specific one
        self.point_digitizer.set_default_context_menu(False)
        
        # Setup star-specific callbacks
        self._setup_callbacks()
        
        # Track selected star for context menu operations
        self._selected_star_index = None
        
    def _setup_callbacks(self):
        """Setup all star-specific callbacks for the point digitizer."""
        self.point_digitizer.set_callback('get_point_data', self._create_star_reference)
        self.point_digitizer.set_callback('on_point_added', self._on_star_added)
        self.point_digitizer.set_callback('on_point_moved', self._on_star_moved)
        self.point_digitizer.set_callback('on_point_removed', self._on_star_removed)
        self.point_digitizer.set_callback('on_context_menu', self._show_star_context_menu)
        self.point_digitizer.set_callback('on_point_selected', self._on_star_selected)
        self.point_digitizer.set_callback('on_digitization_stop', self._save_stars)
        self.point_digitizer.set_callback('get_precise_position', self._get_precise_position)
    
    def start_digitization(self):
        """Start star digitization mode."""

        self._load_existing_stars()
        self.point_digitizer.start_digitization()
        print("Star digitization mode started - left-click to select/add stars, drag to move, right-click for options")
    
    def stop_digitization(self):
        """Stop star digitization mode."""
        self.point_digitizer.stop_digitization()
    
    def set_cloudimage(self, cloudimage):
        """Set the current cloud image."""
        self.cloudimage = cloudimage
        # Load existing stars from cloudimage
        self._load_existing_stars()
    
    def _load_existing_stars(self):
        """Load existing stars from cloudimage into the point digitizer."""
        if self.cloudimage is None:
            return
            
        # Clear existing points
        self.point_digitizer.clear_all_points()
        
        # Add each star as a point
        for star_ref in self.cloudimage.starReferences:
            x, y = star_ref.pixelcoords
            self.point_digitizer.add_point(x, y, star_ref.name, data=star_ref)
    
    def _get_star_name(self, x, y):
        """Get star name from user input."""
        return gui_star_input('Ievadi zvaigznes vārdu')
    
    def _create_star_reference(self, x, y):
        """Create a StarReference object with complex validation logic."""
        input_text = self._get_star_name(x, y)
        
        if input_text is not None and self.cloudimage is not None:
            # Create StarReference with automatic coordinate parsing
            sr = StarReference(input_text, [x, y])

            # Handle different coordinate types
            Ok = False
            if sr.hasDirectAltAz():
                # Alt-Az coordinates were parsed
                Ok = True
                print(f"Pievienota zvaigzne ar Alt-Az koordinātēm: {sr.name}")
            elif sr.skycoord is not None:
                # RA/DEC coordinates were parsed
                Ok = True
                print(f"Pievienota zvaigzne ar RA/DEC koordinātēm: {sr.name}")
            else:
                # Traditional star name - try to resolve
                try:
                    sr.getSkyCoord()
                    Ok = True
                    print(f"Atpazīta zvaigzne: {sr.name}")
                except Exception as e:
                    print(f"Nevarēja atpazīt zvaigzni: {e}")
                    from qthelper import gui_string
                    Ok = gui_string(caption='Neatpazīst zvaigzni, vai ievadīt?') is not None
                    if Ok:
                        print(f"Pievienota neatpazīta zvaigzne: {sr.name}")
            
            if Ok:
                return sr
        
        return None
    
    def _on_star_added(self, point_index, point):
        """Handle when a star is added to the digitizer."""
        if self.cloudimage is not None:
            star_ref = point['data']
            star_ref.pixelcoords = [point['x'], point['y']]
            # Add to star collection if not already there
            print(f'Star added at pixel coords {point}, data: {star_ref}')
            if star_ref not in self.cloudimage.starReferences:
                self.cloudimage.starReferences.append(star_ref)
    
    def _on_star_moved(self, point_index, point, old_x, old_y, new_x, new_y):
        """Handle when a star is moved."""
        star_ref = point['data']
        if star_ref:
            # Update StarReference coordinates
            star_ref.pixelcoords = np.array([new_x, new_y])
            print(f"Moved star '{star_ref.name}' to ({new_x:.1f}, {new_y:.1f})")
    
    def _on_star_removed(self, point_index, point):
        """Handle when a star is removed."""
        if self.cloudimage is not None:
            star_ref = point['data']
            if star_ref in self.cloudimage.starReferences:
                self.cloudimage.starReferences.remove(star_ref)
                print(f"Removed star: {star_ref.name}")
    
    def _on_star_selected(self, point_index, point):
        """Handle when a star is selected."""
        self._selected_star_index = point_index
    
    def _show_star_context_menu(self, point_index, point, event):
        """Show PyQt5 context menu for star operations."""
        self._selected_star_index = point_index
        
        try:
            from PyQt5.QtWidgets import QMenu, QAction
            from PyQt5.QtCore import QPoint
            
            # Create context menu
            menu = QMenu()
            
            star_ref = point['data']
            star_name = star_ref.name if star_ref else point['name']
            
            # Add menu actions
            edit_action = QAction("Edit Name/Coordinates", menu)
            edit_action.triggered.connect(lambda: self._edit_star_coordinates(point_index))
            menu.addAction(edit_action)
            
            pixel_action = QAction("Change Pixel Position", menu)
            pixel_action.triggered.connect(lambda: self._edit_star_pixel_position(point_index))
            menu.addAction(pixel_action)
            
            menu.addSeparator()
            
            delete_action = QAction("Delete Star", menu)
            delete_action.triggered.connect(lambda: self._delete_star(point_index))
            menu.addAction(delete_action)
            
            # Convert matplotlib event coordinates to Qt coordinates
            ax = self.point_digitizer.ax
            canvas_widget = ax.figure.canvas
            if hasattr(canvas_widget, 'mapToGlobal'):
                # Convert canvas coordinates to screen coordinates
                canvas_pos = QPoint(int(event.x), int(canvas_widget.height() - event.y))
                global_pos = canvas_widget.mapToGlobal(canvas_pos)
                menu.exec_(global_pos)
            else:
                # Fallback: show menu at cursor position
                menu.exec_()
                
        except ImportError:
            print("PyQt5 not available for context menu")
        except Exception as e:
            print(f"Error showing context menu: {e}")
        
        return True  # Handled externally
    
    def _edit_star_coordinates(self, point_index):
        """Edit the name/coordinates of the selected star."""
        if not self.point_digitizer._index_valid(point_index):
            return
            
        point = self.point_digitizer._points[point_index]
        star_ref = point['data']
        
        if not star_ref:
            return
        
        # Create input text showing current value
        if star_ref.hasDirectAltAz():
            current_text = f"{star_ref.altaz_coord.az.deg:.1f},{star_ref.altaz_coord.alt.deg:.1f}"
        elif star_ref.skycoord is not None:
            current_text = f"ra:{star_ref.skycoord.ra.deg:.3f},{star_ref.skycoord.dec.deg:.3f}"
        else:
            current_text = star_ref.name
        
        # Get new input from user
        new_input = gui_star_input(f'Edit star (current: {current_text})')
        
        if new_input is not None and new_input.strip():
            # Create a new StarReference with the new input
            new_star = StarReference(new_input, star_ref.pixelcoords.copy())
            
            # Replace the old star in the collection
            if self.cloudimage is not None:
                old_index = self.cloudimage.starReferences.index(star_ref)
                self.cloudimage.starReferences[old_index] = new_star
                
                # Update point data
                point['data'] = new_star
                point['name'] = new_star.name
                
                # Update visual representation
                self.point_digitizer.update_name(point_index, new_star.name)
                
                # Update position if coordinates changed
                x, y = new_star.pixelcoords
                if x != point['x'] or y != point['y']:
                    self.point_digitizer.update_point_position(point_index, x, y)
                
                print(f"Updated star: {new_star.name}")
    
    def _edit_star_pixel_position(self, point_index):
        """Edit the pixel position of the selected star."""
        if not self.point_digitizer._index_valid(point_index):
            return
            
        point = self.point_digitizer._points[point_index]
        star_ref = point['data']
        
        if not star_ref:
            return
            
        current_x, current_y = star_ref.pixelcoords
        
        from qthelper import gui_string
        
        # Get new pixel coordinates
        coord_input = gui_string(
            text=f"{current_x:.1f},{current_y:.1f}",
            caption=f"Enter new pixel coordinates for '{star_ref.name}' (x,y)"
        )
        
        if coord_input is not None:
            try:
                parts = coord_input.split(',')
                if len(parts) == 2:
                    new_x = float(parts[0].strip())
                    new_y = float(parts[1].strip())
                    
                    # Validate coordinates are within reasonable bounds
                    if self.cloudimage is not None:
                        height, width = self.cloudimage.imagearray.shape[:2]
                        if 0 <= new_x < width and 0 <= new_y < height:
                            # Update the star's pixel coordinates
                            star_ref.pixelcoords = np.array([new_x, new_y])
                            
                            # Update point digitizer position
                            self.point_digitizer.update_point_position(point_index, new_x, new_y)
                            
                            print(f"Moved star '{star_ref.name}' to ({new_x:.1f}, {new_y:.1f})")
                        else:
                            print(f"Coordinates ({new_x}, {new_y}) are outside image bounds ({width}x{height})")
                    else:
                        # No bounds checking if no cloudimage
                        star_ref.pixelcoords = np.array([new_x, new_y])
                        self.point_digitizer.update_point_position(point_index, new_x, new_y)
                        print(f"Moved star '{star_ref.name}' to ({new_x:.1f}, {new_y:.1f})")
                else:
                    print("Invalid coordinate format. Use: x,y")
            except ValueError as e:
                print(f"Invalid coordinates: {e}")
    
    def _delete_star(self, point_index):
        """Delete the selected star with confirmation."""
        if not self.point_digitizer._index_valid(point_index):
            return
            
        point = self.point_digitizer._points[point_index]
        star_ref = point['data']
        
        if not star_ref:
            return
            
        star_name = star_ref.name
        
        from qthelper import gui_confirm
        
        # Confirm deletion
        confirm = gui_confirm(
            caption=f"Delete star '{star_name}'?"
        )
        
        if confirm:  # User clicked Yes
            # Remove from point digitizer (this will trigger _on_star_removed)
            self.point_digitizer.remove_point(point_index)
            print(f"Deleted star: {star_name}")
    
    def _save_stars(self, points):
        """Save stars to file when digitization stops."""
        self.main_window.isDigitizeStars = None
        if self.cloudimage is not None:
            from qthelper import gui_save_fname
            import os
            
            filename_stars = os.path.splitext(self.cloudimage.filename)[0]+'_zvaigznes.txt'
            filename_stars = gui_save_fname(
                directory=filename_stars,
                caption='Zvaigžņu fails',
                filter='(*.txt)')
            
            if filename_stars != '':            
                self.cloudimage.saveStarReferences(filename_stars)
                print(f"Saved {len(self.cloudimage.starReferences)} stars to {filename_stars}")
        self.main_window.DrawImage()
        
    def _get_precise_position(self, x, y, ctrl_pressed):
        """
        Get precise position using PSF fitting if ctrl_pressed is True.
        
        Args:
            x, y: Original coordinates
            ctrl_pressed: If Ctrl key was not pressed: Automatic positioning to star position, else exact click positioning
            
        Returns:
            (new_x, new_y) if automatic positioning is requested, None otherwise
        """
        if ctrl_pressed:
            return None
                       
        # Use PSF fitting for precise positioning
        return self._psf_fit_position(x, y)
    
    def _centroid_position(self, x, y):
        """
        Calculate precise position using centroiding algorithm.
        
        Args:
            x, y: Original coordinates (float)
            
        Returns:
            (new_x, new_y): Precise coordinates using centroiding
        """
        if self.cloudimage is None or self.cloudimage.imagearray is None:
            return x, y
            
        subimage, x_min, y_min = self._get_subimage(x,y,self.cloudimage.imagearray)   

        # Calculate centroid
        if subimage.size == 0:
            return x, y
            
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:subimage.shape[0], 0:subimage.shape[1]]
        
        # Calculate weighted centroid
        total_intensity = np.sum(subimage)
        if total_intensity == 0:
            return x, y
            
        centroid_x = np.sum(x_coords * subimage) / total_intensity + x_min
        centroid_y = np.sum(y_coords * subimage) / total_intensity + y_min

        #print(f'Centroid {centroid_x} {centroid_y}, {x} {y}')
        #self.point_digitizer.ax.plot([x_min,x_max,x_max,x_min],[y_min,y_min,y_max,y_max], lw=2, color='red')
        #print(f'{total_intensity=} {image.shape=} {y_coords=} {x_coords=}')
        #print(f'{subimage=}')
        #print(f'{subimage.min()=} {subimage.max()=}')
        #np.save('subimage.npy',subimage)
        return centroid_x, centroid_y
    
    def _get_subimage(self, x, y, image):
        height, width = image.shape[:2]
        
        # Convert to integer pixel coordinates
        center_x = int(round(x))
        center_y = int(round(y))
        
        # Define the box around the original position
        half_box = self.centroid_box_size // 2
        
        # Calculate box boundaries with image bounds checking
        x_min = max(0, center_x - half_box)
        x_max = min(width, center_x + half_box + 1)
        y_min = max(0, center_y - half_box)
        y_max = min(height, center_y + half_box + 1)
        
        # Extract the subimage
        if len(image.shape) == 3:
            # Color image - use luminance
            subimage = image[y_min:y_max, x_min:x_max]
            # Convert to grayscale using luminance formula
            subimage = 0.299 * subimage[:,:,0] + 0.587 * subimage[:,:,1] + 0.114 * subimage[:,:,2]
        else:
            # Grayscale image
            subimage = image[y_min:y_max, x_min:x_max].astype(float)
        return subimage, x_min, y_min

    def _psf_fit_position(self, x, y, sigma=3, fit_shape=(7,7)):
        """
        Calculate precise position using 2D Gaussian PSF fitting.
        
        Args:
            x, y: Original coordinates (float)
            sigma: Background threshold in standard deviations (default: 3)
            
        Returns:
            (new_x, new_y): Precise coordinates using PSF fitting, or original (x, y) if fitting fails
        """
        if self.cloudimage is None or self.cloudimage.imagearray is None:
            return x, y
            
        try:
            import photutils
            import photutils.psf
        except ImportError:
            print("photutils not available, falling back to original coordinates")
            return x, y
        
        subimage, x_min, y_min = self._get_subimage(x,y,self.cloudimage.imagearray)   
        
        # Check if subimage is valid
        if subimage.size == 0:
            return x, y
            
        try:
            # Background subtraction using your method
            s = subimage.std()
            background = subimage.mean() + sigma * s
            subimage = subimage - background
            subimage[subimage < 0] = 0.0
            
            # Check if there's any signal left after background subtraction
            if subimage.max() == 0:
                return x, y
            
            # Fit shape calculation
            #fit_shape = 2 * ((np.array(subimage.shape) + 1) // 2) - 1
                        
            # Perform PSF fitting
            fit_result = photutils.psf.fit_2dgaussian(subimage, fit_shape=fit_shape)
            
            # Check if fitting was successful
            if fit_result is None or not hasattr(fit_result, 'results'):
                return x, y
                
            result = fit_result.results
            result.sort('flux_fit')
            result = result[-1]
            
            # Extract fitted coordinates
            fit_x = result['x_fit']
            fit_y = result['y_fit']
            
            # Check if fitted coordinates are reasonable (within the subimage)
            if (0 <= fit_x < subimage.shape[1] and 
                0 <= fit_y < subimage.shape[0]):
                
                # Convert back to image coordinates
                precise_x = x_min + fit_x
                precise_y = y_min + fit_y
                
                return precise_x, precise_y
            else:
                return x, y
                
        except Exception as e:
            print(f"PSF fitting failed: {e}")
            return x, y
