"""
Camera Parameters Modification Dialog

This module provides a GUI dialog for modifying existing camera parameters
including focal lengths, center position, orientation, distortion parameters
and projection type.

Author: Generated for sudrabainiemakoni project
"""

import sys
import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
from typing import Optional, Tuple


class CameraModificationDialog(QtWidgets.QDialog):
    """
    Dialog for modifying camera parameters directly.
    
    This dialog provides a user-friendly interface for editing:
    - Focal lengths X and Y (in pixels)
    - Center position X and Y (in pixels)
    - Camera orientation: azimuth, elevation, rotation (in degrees)
    - Distortion parameters K1, K2, K3
    - Projection type (rectilinear, equirectangular, stereographic)
    """
    
    def __init__(self, parent=None, camera=None, cloudimage=None):
        """
        Initialize the camera modification dialog.
        
        Args:
            parent: Parent widget
            camera: Camera object with existing parameters to modify (None for new camera)
            cloudimage: CloudImage object (required for creating new cameras)
        """
        super().__init__(parent)
        
        # Load the UI file
        ui_file = os.path.join(os.path.dirname(__file__), 'camera_modification.ui')
        if not os.path.exists(ui_file):
            raise FileNotFoundError(f"UI file not found: {ui_file}")
        
        uic.loadUi(ui_file, self)
        
        # Store references
        self.camera = camera
        self.cloudimage = cloudimage
        self.original_params = {}
        
        # Flag to prevent recursive updates during conversion
        self._updating_focal_length = False
        # Flag to prevent recursive updates during focal length locking
        self._updating_locked_focal = False
        
        # Connect signals
        self.setup_connections()
        
        # Load current camera parameters into UI or set defaults
        self.load_camera_params_to_ui()
        
        # Setup tooltips
        self.setup_tooltips()
        
        # Set window properties
        self.setModal(True)
        self.setWindowTitle("Modify Camera Parameters")
    
    def setup_connections(self):
        """Setup signal connections for UI elements"""
        # Reset button
        self.pushButton_reset.clicked.connect(self.reset_to_current)
        
        # Focal length conversion signals
        self.doubleSpinBox_fx.valueChanged.connect(self.on_fx_px_changed)
        self.doubleSpinBox_fy.valueChanged.connect(self.on_fy_px_changed)
        self.doubleSpinBox_fx_mm.valueChanged.connect(self.on_fx_mm_changed)
        self.doubleSpinBox_fy_mm.valueChanged.connect(self.on_fy_mm_changed)
        
        # Focal length locking checkbox
        self.checkBox_lock_focal.stateChanged.connect(self.on_focal_lock_changed)
    
    def setup_tooltips(self):
        """Setup helpful tooltips for UI elements"""
        self.doubleSpinBox_fx.setToolTip("Focal length in X direction (pixels)")
        self.doubleSpinBox_fy.setToolTip("Focal length in Y direction (pixels)")
        self.doubleSpinBox_cx.setToolTip("Camera center X position (pixels)")
        self.doubleSpinBox_cy.setToolTip("Camera center Y position (pixels)")
        self.doubleSpinBox_azimuth.setToolTip("Camera pointing azimuth (degrees, 0=North)")
        self.doubleSpinBox_elevation.setToolTip("Camera elevation angle (degrees, 0=horizon, 90=zenith)")
        self.doubleSpinBox_rotation.setToolTip("Camera roll rotation (degrees)")
        self.doubleSpinBox_k1.setToolTip("First-order radial distortion coefficient")
        self.doubleSpinBox_k2.setToolTip("Second-order radial distortion coefficient")
        self.doubleSpinBox_k3.setToolTip("Third-order radial distortion coefficient")
        self.comboBox_projection.setToolTip(
            "Camera projection model:\n"
            "• Rectilinear: Standard perspective cameras\n"
            "• Equirectangular: 360° panoramic cameras\n"
            "• Stereographic: Wide-angle fisheye cameras"
        )
        self.doubleSpinBox_fx_mm.setToolTip("Focal length X in 35mm equivalent (automatically converts to/from pixels)")
        self.doubleSpinBox_fy_mm.setToolTip("Focal length Y in 35mm equivalent (automatically converts to/from pixels)")
        self.checkBox_lock_focal.setToolTip("When checked, changing X focal length also updates Y focal length (and vice versa)")
        
        # Update window title based on mode
        if self.camera is None:
            self.setWindowTitle("Create Camera from Manual Parameters")
        else:
            self.setWindowTitle("Modify Camera Parameters")
    
    def load_camera_params_to_ui(self):
        """Load current camera parameters into UI controls"""
        if self.camera is None:
            # Set reasonable defaults for new camera creation
            self.set_default_values()
            return
        
        try:
            # Get focal lengths and center (in pixels)
            fx = self.camera.camera_enu.focallength_x_px
            fy = self.camera.camera_enu.focallength_y_px
            cx = self.camera.camera_enu.center_x_px
            cy = self.camera.camera_enu.center_y_px
            
            # Store original parameters
            self.original_params = {
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
            }
            
            # Set focal lengths - temporarily disable conversion to avoid recursion
            self._updating_focal_length = True
            try:
                self.doubleSpinBox_fx.setValue(fx)
                self.doubleSpinBox_fy.setValue(fy)
                
                # Set corresponding mm values
                image_width, _ = self.get_image_dimensions()
                fx_mm = self.px_to_mm(fx, image_width)
                fy_mm = self.px_to_mm(fy, image_width)
                self.doubleSpinBox_fx_mm.setValue(fx_mm)
                self.doubleSpinBox_fy_mm.setValue(fy_mm)
            finally:
                self._updating_focal_length = False
            
            # Set center position
            self.doubleSpinBox_cx.setValue(cx)
            self.doubleSpinBox_cy.setValue(cy)
            
            # Get orientation parameters
            az, el, rot = self.camera.get_azimuth_elevation_rotation()
            self.original_params.update({'az': az, 'el': el, 'rot': rot})
            
            self.doubleSpinBox_azimuth.setValue(az)
            self.doubleSpinBox_elevation.setValue(el)
            self.doubleSpinBox_rotation.setValue(rot)
            
            # Get distortion parameters directly from camera_enu
            k1 = getattr(self.camera.camera_enu, 'k1', 0.0)
            k2 = getattr(self.camera.camera_enu, 'k2', 0.0)
            k3 = getattr(self.camera.camera_enu, 'k3', 0.0)
            
            self.original_params.update({'k1': k1, 'k2': k2, 'k3': k3})
            
            self.doubleSpinBox_k1.setValue(k1)
            self.doubleSpinBox_k2.setValue(k2)
            self.doubleSpinBox_k3.setValue(k3)
            
            # Get projection type
            projection_name = type(self.camera.camera_enu.projection).__name__.lower()
            if 'rectilinear' in projection_name:
                self.comboBox_projection.setCurrentIndex(0)
                self.original_params['projection'] = 'rectilinear'
            elif 'equirectangular' in projection_name:
                self.comboBox_projection.setCurrentIndex(1)
                self.original_params['projection'] = 'equirectangular'
            elif 'stereographic' in projection_name:
                self.comboBox_projection.setCurrentIndex(2)
                self.original_params['projection'] = 'stereographic'
            else:
                self.comboBox_projection.setCurrentIndex(0)  # Default to rectilinear
                self.original_params['projection'] = 'rectilinear'
                
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            # Set default values if loading fails
            self.set_default_values()
    
    def set_default_values(self):
        """Set reasonable default values based on image dimensions"""
        # Default fallback values
        image_width = 1024
        image_height = 768
        
        # Get actual image dimensions if cloudimage is available
        if self.cloudimage is not None and hasattr(self.cloudimage, 'imagearray'):
            image_height, image_width = self.cloudimage.imagearray.shape[:2]
        
        # Calculate focal length equivalent to 24mm on full frame (36mm sensor width)
        focal_length_px = image_width / 36.0 * 24.0
        
        # Set centers at half image size
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        
        # Temporarily disable conversion to avoid recursion
        self._updating_focal_length = True
        try:
            self.doubleSpinBox_fx.setValue(focal_length_px)
            self.doubleSpinBox_fy.setValue(focal_length_px)
            
            # Set corresponding mm values
            focal_length_mm = self.px_to_mm(focal_length_px, image_width)
            self.doubleSpinBox_fx_mm.setValue(focal_length_mm)
            self.doubleSpinBox_fy_mm.setValue(focal_length_mm)
        finally:
            self._updating_focal_length = False
            
        self.doubleSpinBox_cx.setValue(center_x)
        self.doubleSpinBox_cy.setValue(center_y)
        self.doubleSpinBox_azimuth.setValue(0.0)  
        self.doubleSpinBox_elevation.setValue(30.0)
        self.doubleSpinBox_rotation.setValue(0.0)
        self.doubleSpinBox_k1.setValue(0.0)
        self.doubleSpinBox_k2.setValue(0.0)
        self.doubleSpinBox_k3.setValue(0.0)
        self.comboBox_projection.setCurrentIndex(0)
    
    def reset_to_current(self):
        """Reset all parameters to current camera values"""
        if self.camera is not None:
            self.load_camera_params_to_ui()
        else:
            self.set_default_values()
    
    def get_image_dimensions(self):
        """Get image dimensions from cloudimage"""
        if self.cloudimage is not None and hasattr(self.cloudimage, 'imagearray'):
            height, width = self.cloudimage.imagearray.shape[:2]
            return width, height
        return 1024, 768  # fallback
    
    def px_to_mm(self, focal_px, image_width):
        """Convert focal length from pixels to 35mm equivalent"""
        return focal_px * 36.0 / image_width
    
    def mm_to_px(self, focal_mm, image_width):
        """Convert focal length from 35mm equivalent to pixels"""
        return focal_mm * image_width / 36.0
    
    def on_fx_px_changed(self, value):
        """Handle focal length X pixel value change"""
        if self._updating_focal_length or self._updating_locked_focal:
            return
        self._updating_focal_length = True
        try:
            image_width, _ = self.get_image_dimensions()
            fx_mm = self.px_to_mm(value, image_width)
            self.doubleSpinBox_fx_mm.setValue(fx_mm)
            
            # If focal lengths are locked, update Y values too
            if self.checkBox_lock_focal.isChecked():
                self._updating_locked_focal = True
                self.doubleSpinBox_fy.setValue(value)
                self.doubleSpinBox_fy_mm.setValue(fx_mm)
                self._updating_locked_focal = False
        finally:
            self._updating_focal_length = False
    
    def on_fy_px_changed(self, value):
        """Handle focal length Y pixel value change"""
        if self._updating_focal_length or self._updating_locked_focal:
            return
        self._updating_focal_length = True
        try:
            image_width, _ = self.get_image_dimensions()
            fy_mm = self.px_to_mm(value, image_width)
            self.doubleSpinBox_fy_mm.setValue(fy_mm)
            
            # If focal lengths are locked, update X values too
            if self.checkBox_lock_focal.isChecked():
                self._updating_locked_focal = True
                self.doubleSpinBox_fx.setValue(value)
                self.doubleSpinBox_fx_mm.setValue(fy_mm)
                self._updating_locked_focal = False
        finally:
            self._updating_focal_length = False
    
    def on_fx_mm_changed(self, value):
        """Handle focal length X mm value change"""
        if self._updating_focal_length or self._updating_locked_focal:
            return
        self._updating_focal_length = True
        try:
            image_width, _ = self.get_image_dimensions()
            fx_px = self.mm_to_px(value, image_width)
            self.doubleSpinBox_fx.setValue(fx_px)
            
            # If focal lengths are locked, update Y values too
            if self.checkBox_lock_focal.isChecked():
                self._updating_locked_focal = True
                self.doubleSpinBox_fy.setValue(fx_px)
                self.doubleSpinBox_fy_mm.setValue(value)
                self._updating_locked_focal = False
        finally:
            self._updating_focal_length = False
    
    def on_fy_mm_changed(self, value):
        """Handle focal length Y mm value change"""
        if self._updating_focal_length or self._updating_locked_focal:
            return
        self._updating_focal_length = True
        try:
            image_width, _ = self.get_image_dimensions()
            fy_px = self.mm_to_px(value, image_width)
            self.doubleSpinBox_fy.setValue(fy_px)
            
            # If focal lengths are locked, update X values too
            if self.checkBox_lock_focal.isChecked():
                self._updating_locked_focal = True
                self.doubleSpinBox_fx.setValue(fy_px)
                self.doubleSpinBox_fx_mm.setValue(value)
                self._updating_locked_focal = False
        finally:
            self._updating_focal_length = False
    
    def on_focal_lock_changed(self, state):
        """Handle focal length lock checkbox state change"""
        if state == 2:  # Qt.Checked
            # When locking is enabled, synchronize Y values to match X values
            if not self._updating_focal_length and not self._updating_locked_focal:
                self._updating_locked_focal = True
                try:
                    fx_value = self.doubleSpinBox_fx.value()
                    fx_mm_value = self.doubleSpinBox_fx_mm.value()
                    self.doubleSpinBox_fy.setValue(fx_value)
                    self.doubleSpinBox_fy_mm.setValue(fx_mm_value)
                finally:
                    self._updating_locked_focal = False
    
    def get_modified_parameters(self) -> dict:
        """
        Get the modified parameters from UI.
        
        Returns:
            Dictionary with all modified camera parameters
        """
        projection_types = ['rectilinear', 'equirectangular', 'stereographic']
        
        return {
            'fx': self.doubleSpinBox_fx.value(),
            'fy': self.doubleSpinBox_fy.value(),
            'cx': self.doubleSpinBox_cx.value(),
            'cy': self.doubleSpinBox_cy.value(),
            'azimuth': self.doubleSpinBox_azimuth.value(),
            'elevation': self.doubleSpinBox_elevation.value(),
            'rotation': self.doubleSpinBox_rotation.value(),
            'k1': self.doubleSpinBox_k1.value(),
            'k2': self.doubleSpinBox_k2.value(),
            'k3': self.doubleSpinBox_k3.value(),
            'projection': projection_types[self.comboBox_projection.currentIndex()]
        }
    
    def apply_parameters_to_camera(self):
        """Apply the modified parameters to the camera object"""
        if self.camera is None:
            # Cannot apply to non-existent camera - this should be handled by create_camera_from_parameters
            return False
        
        try:
            params = self.get_modified_parameters()
            
            # Apply focal lengths and center to camera_enu
            self.camera.camera_enu.focallength_x_px = params['fx']
            self.camera.camera_enu.focallength_y_px = params['fy']
            self.camera.camera_enu.center_x_px = params['cx']
            self.camera.camera_enu.center_y_px = params['cy']
            
            # Apply orientation
            # Convert elevation back to tilt (tilt = elevation + 90)
            tilt = params['elevation'] + 90
            
            self.camera.camera_enu.heading_deg = params['azimuth']
            self.camera.camera_enu.tilt_deg = tilt
            self.camera.camera_enu.roll_deg = params['rotation']
            
            # Apply distortion parameters directly to camera_enu
            self.camera.camera_enu.k1 = params['k1']
            self.camera.camera_enu.k2 = params['k2']  
            self.camera.camera_enu.k3 = params['k3']
            
            # Apply projection type (requires recreating camera with new projection)
            if params['projection'] != self.original_params.get('projection', 'rectilinear'):
                print(f"Note: Projection type change to {params['projection']} requires camera recalibration")
            
            # Update the ECEF camera using the proper method
            self.camera.camera_ecef = self.camera.camera_ecef_from_camera_enu(self.camera.camera_enu)

            return True
            
        except Exception as e:
            print(f"Error applying camera parameters: {e}")
            return False
    
    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate current parameter settings.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        params = self.get_modified_parameters()
        
        # Validate focal lengths
        if params['fx'] <= 0 or params['fy'] <= 0:
            return False, "Focal lengths must be positive"
        
        # Validate reasonable ranges
        if params['fx'] < 10 or params['fx'] > 50000:
            return False, "Focal length X seems unreasonable (10-50000 pixels expected)"
        
        if params['fy'] < 10 or params['fy'] > 50000:
            return False, "Focal length Y seems unreasonable (10-50000 pixels expected)"
        
        # Validate elevation range
        if params['elevation'] < -90 or params['elevation'] > 90:
            return False, "Elevation must be between -90 and 90 degrees"
        
        # Validate distortion coefficients (reasonable ranges)
        for k_name, k_val in [('k1', params['k1']), ('k2', params['k2']), ('k3', params['k3'])]:
            if abs(k_val) > 1.0:
                return False, f"Distortion coefficient {k_name} seems too large (|{k_name}| > 1.0)"
        
        return True, ""
    
    def accept(self):
        """Handle OK button click with validation"""
        is_valid, error_msg = self.validate_parameters()
        
        if not is_valid:
            QMessageBox.warning(
                self,
                "Invalid Parameters", 
                f"Parameter validation failed:\n\n{error_msg}"
            )
            return
        
        # Handle camera creation or modification
        if self.camera is None:
            # Create new camera
            if self.cloudimage is None:
                QMessageBox.warning(
                    self,
                    "Error",
                    "CloudImage is required to create a new camera"
                )
                return
                
            new_camera = self.create_camera_from_parameters(self.cloudimage)
            if new_camera is None:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to create camera from parameters"
                )
                return
            
            # Assign the new camera to cloudimage
            self.cloudimage.camera = new_camera
            print('Izveidota jauna kamera no manuālajiem parametriem')
            
        else:
            # Modify existing camera
            success = self.apply_parameters_to_camera()
            if not success:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to apply parameters to camera object"
                )
                return
            print('Kameras parametri modificēti')
        
        super().accept()
    
    def create_camera_from_parameters(self, cloudImage):
        """
        Create a new camera from the dialog parameters.
        
        Args:
            cloudImage: CloudImage object needed for camera creation
            
        Returns:
            Camera object created from manual parameters, or None if failed
        """
        try:
            from sudrabainiemakoni.cloudimage_camera import Camera
            
            params = self.get_modified_parameters()
            
            camera = Camera.from_manual_parameters(
                cloudImage=cloudImage,
                fx=params['fx'],
                fy=params['fy'], 
                cx=params['cx'],
                cy=params['cy'],
                azimuth=params['azimuth'],
                elevation=params['elevation'],
                rotation=params['rotation'],
                k1=params['k1'],
                k2=params['k2'],
                k3=params['k3'],
                projection=params['projection']
            )
            
            return camera
            
        except Exception as e:
            print(f"Error creating camera from parameters: {e}")
            return None


def show_camera_modification_dialog(parent=None, camera=None, cloudimage=None) -> Tuple[bool, Optional[dict]]:
    """
    Convenience function to show the camera modification dialog.
    
    Args:
        parent: Parent widget
        camera: Camera object to modify (None for new camera creation)
        cloudimage: CloudImage object (required for new camera creation)
    
    Returns:
        Tuple of (dialog_accepted, modified_parameters_dict)
    """
    dialog = CameraModificationDialog(parent, camera, cloudimage)
    accepted = dialog.exec_() == QtWidgets.QDialog.Accepted
    
    if accepted:
        return True, dialog.get_modified_parameters()
    else:
        return False, None


# Test the dialog when run directly
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Test with no camera (default values)
    accepted, result = show_camera_modification_dialog()
    
    if accepted:
        print("Dialog accepted!")
        print(f"Modified parameters: {result}")
    else:
        print("Dialog cancelled")
    
    sys.exit()