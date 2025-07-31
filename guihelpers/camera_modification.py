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
    
    def __init__(self, parent=None, camera=None):
        """
        Initialize the camera modification dialog.
        
        Args:
            parent: Parent widget
            camera: Camera object with existing parameters to modify
        """
        super().__init__(parent)
        
        # Load the UI file
        ui_file = os.path.join(os.path.dirname(__file__), 'camera_modification.ui')
        if not os.path.exists(ui_file):
            raise FileNotFoundError(f"UI file not found: {ui_file}")
        
        uic.loadUi(ui_file, self)
        
        # Store camera reference
        self.camera = camera
        self.original_params = {}
        
        # Connect signals
        self.setup_connections()
        
        # Load current camera parameters into UI
        if camera is not None:
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
    
    def load_camera_params_to_ui(self):
        """Load current camera parameters into UI controls"""
        if self.camera is None:
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
            
            # Set focal lengths
            self.doubleSpinBox_fx.setValue(fx)
            self.doubleSpinBox_fy.setValue(fy)
            
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
        """Set reasonable default values"""
        self.doubleSpinBox_fx.setValue(1000.0)
        self.doubleSpinBox_fy.setValue(1000.0)
        self.doubleSpinBox_cx.setValue(512.0)
        self.doubleSpinBox_cy.setValue(384.0)
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
        
        # Apply parameters to camera
        if self.camera is not None:
            success = self.apply_parameters_to_camera()
            if not success:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to apply parameters to camera object"
                )
                return
        
        super().accept()


def show_camera_modification_dialog(parent=None, camera=None) -> Tuple[bool, Optional[dict]]:
    """
    Convenience function to show the camera modification dialog.
    
    Args:
        parent: Parent widget
        camera: Camera object to modify
    
    Returns:
        Tuple of (dialog_accepted, modified_parameters_dict)
    """
    dialog = CameraModificationDialog(parent, camera)
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