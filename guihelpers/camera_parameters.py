"""
Camera Parameters Dialog

This module provides a GUI dialog for configuring camera calibration parameters
using the CameraCalibrationParams dataclass. The dialog is designed with Qt Designer
and provides intuitive controls for all camera fitting parameters.

Author: Generated for sudrabainiemakoni project
"""

import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import os

# Import the parameter classes
from sudrabainiemakoni.cloudimage_camera import CameraCalibrationParams, DistortionOrder


class CameraParametersDialog(QtWidgets.QDialog):
    """
    Dialog for editing camera calibration parameters.
    
    This dialog provides a user-friendly interface for configuring:
    - Lens distortion correction order (0-3)
    - Camera projection model (rectilinear, equirectangular, stereographic)
    - Optimization options (centers, separate X/Y focal lengths)
    """
    
    def __init__(self, parent=None, initial_params=None):
        """
        Initialize the camera parameters dialog.
        
        Args:
            parent: Parent widget
            initial_params: CameraCalibrationParams or dict with initial values
        """
        super().__init__(parent)
        
        # Load the UI file
        ui_file = os.path.join(os.path.dirname(__file__), 'camera_parameters.ui')
        if not os.path.exists(ui_file):
            raise FileNotFoundError(f"UI file not found: {ui_file}")
        
        uic.loadUi(ui_file, self)
        
        # Initialize parameters
        if initial_params is None:
            self.params = CameraCalibrationParams()
        elif isinstance(initial_params, dict):
            self.params = CameraCalibrationParams.from_dict(initial_params)
        elif isinstance(initial_params, CameraCalibrationParams):
            self.params = initial_params
        else:
            raise ValueError(f"initial_params must be CameraCalibrationParams or dict, got {type(initial_params)}")
        
        # Connect signals
        self.setup_connections()
        
        # Load initial values into UI
        self.load_params_to_ui()
        
        # Set window properties
        self.setModal(True)
        self.setWindowTitle("Camera Calibration Parameters")
    
    def setup_connections(self):
        """Setup signal connections for UI elements"""
        # Reset button
        self.pushButton_reset.clicked.connect(self.reset_to_defaults)
        
        # Help tooltips
        self.setup_tooltips()
    
    def setup_tooltips(self):
        """Setup helpful tooltips for UI elements"""
        self.comboBox_distortion.setToolTip(
            "Higher order corrections handle more complex lens distortions\n"
            "but may overfit with insufficient reference points.\n\n"
            "Order 4 includes tangential distortion (p1, p2) which accounts\n"
            "for lens elements not being perfectly parallel.\n\n"
            "Order 5 adds rational distortion (k4, k5, k6) for complex radial\n"
            "distortions in high-precision applications."
        )

        self.comboBox_projection.setToolTip(
            "Choose projection model based on your camera/lens type:\n"
            "• Rectilinear: Normal perspective cameras\n"
            "• Equirectangular: 360° panoramic images\n"
            "• Stereographic: Wide-angle fisheye lenses"
        )

        self.groupBox_rotation.setToolTip(
            "Enable to optimize camera orientation (heading, tilt, roll).\n"
            "Disable to keep current rotation fixed."
        )

        self.groupBox_intrinsics.setToolTip(
            "Enable to optimize camera intrinsic parameters (focal length, centers).\n"
            "Disable to keep current intrinsics fixed."
        )

        self.groupBox_distortion_opt.setToolTip(
            "Enable to optimize lens distortion coefficients.\n"
            "Disable to keep existing distortion values unchanged."
        )

        self.checkBox_focallength.setToolTip(
            "Allow optimization of camera focal length.\n"
            "Usually recommended unless you have precise calibration data."
        )

        self.checkBox_centers.setToolTip(
            "Allow optimization of camera center position (principal point).\n"
            "Usually recommended unless you have precise calibration data."
        )

        self.checkBox_separate_xy.setToolTip(
            "Use different focal lengths for X and Y axes.\n"
            "Recommended for most cameras to account for sensor variations."
        )
    
    def load_params_to_ui(self):
        """Load parameter values into UI controls"""
        # Store the actual distortion order value (for when optimization is disabled)
        if self.params.distortion is not None:
            self._last_distortion_order = int(self.params.distortion)
            self.comboBox_distortion.setCurrentIndex(self._last_distortion_order)
        else:
            # If distortion is None, use previously saved order or default to 3
            if not hasattr(self, '_last_distortion_order'):
                self._last_distortion_order = 3  # Default to third order
            self.comboBox_distortion.setCurrentIndex(self._last_distortion_order)

        # Projection type
        projection_map = {
            'rectilinear': 0,
            'equirectangular': 1,
            'stereographic': 2
        }
        self.comboBox_projection.setCurrentIndex(
            projection_map.get(self.params.projectiontype, 0)
        )

        # Optimization group checkboxes
        self.groupBox_rotation.setChecked(self.params.optimize_rotation)
        self.groupBox_intrinsics.setChecked(self.params.optimize_focallength or self.params.centers)
        self.groupBox_distortion_opt.setChecked(self.params.distortion is not None)

        # Individual optimization options
        self.checkBox_focallength.setChecked(self.params.optimize_focallength)
        self.checkBox_centers.setChecked(self.params.centers)
        self.checkBox_separate_xy.setChecked(self.params.separate_x_y)
    
    def save_params_from_ui(self):
        """Save UI values back to parameters object"""
        # Always save the current distortion order selection
        self._last_distortion_order = self.comboBox_distortion.currentIndex()

        # Distortion order - set to None if distortion optimization is disabled
        if self.groupBox_distortion_opt.isChecked():
            self.params.distortion = DistortionOrder(self._last_distortion_order)
        else:
            self.params.distortion = None

        # Projection type
        projection_types = ['rectilinear', 'equirectangular', 'stereographic']
        self.params.projectiontype = projection_types[self.comboBox_projection.currentIndex()]

        # Optimization group options
        self.params.optimize_rotation = self.groupBox_rotation.isChecked()

        # Individual intrinsics options (only matter if intrinsics group is checked)
        if self.groupBox_intrinsics.isChecked():
            self.params.optimize_focallength = self.checkBox_focallength.isChecked()
            self.params.centers = self.checkBox_centers.isChecked()
            self.params.separate_x_y = self.checkBox_separate_xy.isChecked()
        else:
            # When intrinsics group is unchecked, disable all intrinsics optimization
            self.params.optimize_focallength = False
            self.params.centers = False
            self.params.separate_x_y = False
    
    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        self.params = CameraCalibrationParams()  # Create with defaults
        self.load_params_to_ui()
    
    def validate_parameters(self) -> tuple[bool, str]:
        """
        Validate current parameter settings.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Save current UI state to params
        self.save_params_from_ui()
        
        # Validate using the dataclass method
        return self.params.validate()
    
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
        
        # Save final parameters
        self.save_params_from_ui()
        super().accept()
    
    def get_parameters(self) -> CameraCalibrationParams:
        """
        Get the configured parameters.
        
        Returns:
            CameraCalibrationParams object with current settings
        """
        return self.params
    
    def get_parameters_dict(self) -> dict:
        """
        Get parameters as dictionary for backward compatibility.
        
        Returns:
            Dictionary suitable for **kwargs unpacking
        """
        return self.params.to_dict()


def show_camera_parameters_dialog(parent=None, initial_params=None) -> tuple[bool, CameraCalibrationParams]:
    """
    Convenience function to show the camera parameters dialog.
    
    Args:
        parent: Parent widget
        initial_params: Initial parameter values (CameraCalibrationParams or dict)
    
    Returns:
        Tuple of (dialog_accepted, parameters)
    """
    dialog = CameraParametersDialog(parent, initial_params)
    accepted = dialog.exec_() == QtWidgets.QDialog.Accepted
    
    if accepted:
        return True, dialog.get_parameters()
    else:
        return False, initial_params if isinstance(initial_params, CameraCalibrationParams) else CameraCalibrationParams()


# Test the dialog when run directly
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Test with initial parameters
    test_params = CameraCalibrationParams(
        distortion=DistortionOrder.SECOND_ORDER,
        centers=True,
        separate_x_y=False,
        projectiontype='equirectangular'
    )
    
    accepted, result = show_camera_parameters_dialog(initial_params=test_params)
    
    if accepted:
        print("Dialog accepted!")
        print(f"Parameters: {result}")
        print(f"As dict: {result.to_dict()}")
        print(f"Distortion description: {result.get_distortion_description()}")
        print(f"Projection description: {result.get_projection_description()}")
    else:
        print("Dialog cancelled")
    
    sys.exit()