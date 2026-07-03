"""
Camera Parameters Modification Dialog

This module provides a GUI dialog for modifying existing camera parameters
including focal lengths, center position, orientation, distortion parameters
and projection type.

Author: Generated for sudrabainiemakoni project
"""

import sys
import os
import json
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

        # Rational distortion checkbox
        self.checkBox_use_rational.stateChanged.connect(self.on_rational_checkbox_changed)

        # Tangential distortion checkbox
        self.checkBox_use_tangential.stateChanged.connect(self.on_tangential_checkbox_changed)

        # JSON import/export buttons
        self.pushButton_apply_json.clicked.connect(self.on_apply_json)
        self.pushButton_export_json.clicked.connect(self.on_export_to_json)
    
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
        self.checkBox_use_rational.setToolTip("Enable rational distortion (k4, k5, k6) denominator coefficients")
        self.doubleSpinBox_k4.setToolTip("Fourth-order rational distortion coefficient (denominator)")
        self.doubleSpinBox_k5.setToolTip("Fifth-order rational distortion coefficient (denominator)")
        self.doubleSpinBox_k6.setToolTip("Sixth-order rational distortion coefficient (denominator)")
        self.checkBox_use_tangential.setToolTip("Enable tangential distortion (p1, p2) for lenses with non-parallel elements")
        self.doubleSpinBox_p1.setToolTip("First tangential distortion coefficient")
        self.doubleSpinBox_p2.setToolTip("Second tangential distortion coefficient")
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
            k4 = getattr(self.camera.camera_enu, 'k4', 0.0)
            k5 = getattr(self.camera.camera_enu, 'k5', 0.0)
            k6 = getattr(self.camera.camera_enu, 'k6', 0.0)
            p1 = getattr(self.camera.camera_enu, 'p1', 0.0)
            p2 = getattr(self.camera.camera_enu, 'p2', 0.0)

            self.original_params.update({'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'p1': p1, 'p2': p2})

            self.doubleSpinBox_k1.setValue(k1)
            self.doubleSpinBox_k2.setValue(k2)
            self.doubleSpinBox_k3.setValue(k3)
            self.doubleSpinBox_k4.setValue(k4)
            self.doubleSpinBox_k5.setValue(k5)
            self.doubleSpinBox_k6.setValue(k6)
            self.doubleSpinBox_p1.setValue(p1)
            self.doubleSpinBox_p2.setValue(p2)

            # Set rational checkbox based on whether k4/k5/k6 is non-zero
            use_rational = bool(abs(k4) > 1e-9 or abs(k5) > 1e-9 or abs(k6) > 1e-9) #bool - for newer versions numpy.bool is not accepted
            self.checkBox_use_rational.setChecked(use_rational)

            # Set tangential checkbox based on whether p1 or p2 is non-zero
            use_tangential = bool(abs(p1) > 1e-9 or abs(p2) > 1e-9)
            self.checkBox_use_tangential.setChecked(use_tangential)
            
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
            raise
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
        self.doubleSpinBox_k4.setValue(0.0)
        self.doubleSpinBox_k5.setValue(0.0)
        self.doubleSpinBox_k6.setValue(0.0)
        self.doubleSpinBox_p1.setValue(0.0)
        self.doubleSpinBox_p2.setValue(0.0)
        self.checkBox_use_rational.setChecked(False)
        self.checkBox_use_tangential.setChecked(False)
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

    def on_rational_checkbox_changed(self, state):
        """Handle rational distortion checkbox state change"""
        is_enabled = (state == 2)  # Qt.Checked

        self.label_rational.setEnabled(is_enabled)
        self.label_k4.setEnabled(is_enabled)
        self.label_k5.setEnabled(is_enabled)
        self.label_k6.setEnabled(is_enabled)
        self.doubleSpinBox_k4.setEnabled(is_enabled)
        self.doubleSpinBox_k5.setEnabled(is_enabled)
        self.doubleSpinBox_k6.setEnabled(is_enabled)

        if not is_enabled:
            self.doubleSpinBox_k4.setValue(0.0)
            self.doubleSpinBox_k5.setValue(0.0)
            self.doubleSpinBox_k6.setValue(0.0)

    def on_tangential_checkbox_changed(self, state):
        """Handle tangential distortion checkbox state change"""
        is_enabled = (state == 2)  # Qt.Checked

        # Enable/disable tangential distortion controls
        self.label_tangential.setEnabled(is_enabled)
        self.label_p1.setEnabled(is_enabled)
        self.label_p2.setEnabled(is_enabled)
        self.doubleSpinBox_p1.setEnabled(is_enabled)
        self.doubleSpinBox_p2.setEnabled(is_enabled)

        # Reset values when disabling
        if not is_enabled:
            self.doubleSpinBox_p1.setValue(0.0)
            self.doubleSpinBox_p2.setValue(0.0)

    def on_apply_json(self):
        """Parse JSON from the text area and fill all controls"""
        text = self.plainTextEdit_json.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Input", "Please paste a JSON string first.")
            return

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Invalid JSON", f"Could not parse JSON:\n{e}")
            return

        try:
            self._load_from_json_dict(data)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply JSON parameters:\n{e}")

    def on_export_to_json(self):
        """Fill the JSON textarea with parameters from the current camera"""
        if self.camera is None or self.camera.camera_enu is None:
            QMessageBox.warning(self, "No Camera", "No camera loaded to export.")
            return
        from sudrabainiemakoni.cloudimage_camera import name_by_distortion
        cam = self.camera.camera_enu
        keys = cam.parameters.parameters.keys()
        data = {key: getattr(cam, key) for key in keys}
        data['distortiontype'] = name_by_distortion(cam.lens)
        from sudrabainiemakoni import cameraprojections
        data['projectiontype'] = cameraprojections.name_by_projection(cam.projection)
        self.plainTextEdit_json.setPlainText(json.dumps(data, indent=2))

    def _load_from_json_dict(self, data: dict):
        """Fill controls from a parsed camera JSON dictionary"""
        # Focal lengths
        fx = float(data.get('focallength_x_px', self.doubleSpinBox_fx.value()))
        fy = float(data.get('focallength_y_px', self.doubleSpinBox_fy.value()))

        # Use image_width from JSON for mm conversion if available, otherwise fall back
        image_width_json = data.get('image_width_px')
        if image_width_json:
            image_width = float(image_width_json)
        else:
            image_width, _ = self.get_image_dimensions()

        self._updating_focal_length = True
        try:
            self.doubleSpinBox_fx.setValue(fx)
            self.doubleSpinBox_fy.setValue(fy)
            self.doubleSpinBox_fx_mm.setValue(self.px_to_mm(fx, image_width))
            self.doubleSpinBox_fy_mm.setValue(self.px_to_mm(fy, image_width))
        finally:
            self._updating_focal_length = False

        # Center position
        if 'center_x_px' in data:
            self.doubleSpinBox_cx.setValue(float(data['center_x_px']))
        if 'center_y_px' in data:
            self.doubleSpinBox_cy.setValue(float(data['center_y_px']))

        # Orientation — JSON stores raw tilt_deg/heading_deg/roll_deg
        # Apply same logic as get_azimuth_elevation_rotation in cloudimage_camera.py
        if 'tilt_deg' in data and 'heading_deg' in data and 'roll_deg' in data:
            tilt = float(data['tilt_deg'])
            heading = float(data['heading_deg'])
            roll = float(data['roll_deg'])
            if tilt < 0:
                azimuth = 180 + heading if heading < 0 else heading - 180
                tilt_abs = -tilt
                roll_conv = roll - 180 if roll > 0 else 180 + roll
            else:
                azimuth = heading
                tilt_abs = tilt
                roll_conv = roll
            elevation = tilt_abs - 90
            self.doubleSpinBox_azimuth.setValue(azimuth)
            self.doubleSpinBox_elevation.setValue(elevation)
            self.doubleSpinBox_rotation.setValue(roll_conv)
        else:
            if 'tilt_deg' in data:
                self.doubleSpinBox_elevation.setValue(float(data['tilt_deg']) - 90.0)
            if 'heading_deg' in data:
                self.doubleSpinBox_azimuth.setValue(float(data['heading_deg']))
            if 'roll_deg' in data:
                self.doubleSpinBox_rotation.setValue(float(data['roll_deg']))

        # Distortion k1-k3
        for name in ('k1', 'k2', 'k3'):
            if name in data:
                getattr(self, f'doubleSpinBox_{name}').setValue(float(data[name]))

        # Rational distortion k4-k6
        k4 = float(data.get('k4', 0.0))
        k5 = float(data.get('k5', 0.0))
        k6 = float(data.get('k6', 0.0))
        use_rational = abs(k4) > 1e-9 or abs(k5) > 1e-9 or abs(k6) > 1e-9
        self.checkBox_use_rational.setChecked(use_rational)
        self.doubleSpinBox_k4.setValue(k4)
        self.doubleSpinBox_k5.setValue(k5)
        self.doubleSpinBox_k6.setValue(k6)

        # Tangential distortion p1/p2
        p1 = float(data.get('p1', 0.0))
        p2 = float(data.get('p2', 0.0))
        use_tangential = abs(p1) > 1e-9 or abs(p2) > 1e-9
        self.checkBox_use_tangential.setChecked(use_tangential)
        self.doubleSpinBox_p1.setValue(p1)
        self.doubleSpinBox_p2.setValue(p2)

        # Projection type from projectiontype field
        projection_type = str(data.get('projectiontype', '')).lower()
        if 'equirect' in projection_type:
            self.comboBox_projection.setCurrentIndex(1)
        elif 'stereo' in projection_type:
            self.comboBox_projection.setCurrentIndex(2)
        else:
            self.comboBox_projection.setCurrentIndex(0)

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
            'k4': self.doubleSpinBox_k4.value() if self.checkBox_use_rational.isChecked() else 0.0,
            'k5': self.doubleSpinBox_k5.value() if self.checkBox_use_rational.isChecked() else 0.0,
            'k6': self.doubleSpinBox_k6.value() if self.checkBox_use_rational.isChecked() else 0.0,
            'use_rational': self.checkBox_use_rational.isChecked(),
            'p1': self.doubleSpinBox_p1.value() if self.checkBox_use_tangential.isChecked() else 0.0,
            'p2': self.doubleSpinBox_p2.value() if self.checkBox_use_tangential.isChecked() else 0.0,
            'use_tangential': self.checkBox_use_tangential.isChecked(),
            'projection': projection_types[self.comboBox_projection.currentIndex()]
        }
    
    def apply_parameters_to_camera(self):
        """Apply the modified parameters to the camera object"""
        if self.camera is None:
            # Cannot apply to non-existent camera - this should be handled by create_camera_from_parameters
            return False
        
        try:
            from sudrabainiemakoni.cloudimage_camera import camera_from_dict

            params = self.get_modified_parameters()
            image_size = self.camera.image_size

            if params['use_rational']:
                distortion_type = 'rationaldistortionlimited'
            else:
                distortion_type = 'brownlensdistortionlimited'

            tilt = params['elevation'] + 90
            variables = {
                'focallength_x_px': params['fx'],
                'focallength_y_px': params['fy'],
                'center_x_px': params['cx'],
                'center_y_px': params['cy'],
                'image_width_px': image_size[0],
                'image_height_px': image_size[1],
                'heading_deg': params['azimuth'],
                'tilt_deg': tilt,
                'roll_deg': params['rotation'],
                'pos_x_m': 0.0,
                'pos_y_m': 0.0,
                'elevation_m': 0.0,
                'k1': params['k1'],
                'k2': params['k2'],
                'k3': params['k3'],
                'k4': params['k4'],
                'k5': params['k5'],
                'k6': params['k6'],
                'p1': params['p1'],
                'p2': params['p2'],
                'distortiontype': distortion_type,
                'projectiontype': params['projection'],
            }

            self.camera.camera_enu = camera_from_dict(variables)

            if self.cloudimage is not None:
                self.camera.camera_ecef = self.camera.camera_ecef_from_camera_enu(self.camera.camera_enu, self.cloudimage.location)
            else:
                print("Warning: Cannot update ECEF camera without cloudimage location")

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
        for k_name, k_val in [('k1', params['k1']), ('k2', params['k2']), ('k3', params['k3']),
                               ('k4', params['k4']), ('k5', params['k5']), ('k6', params['k6'])]:
            if abs(k_val) > 1.0:
                return False, f"Distortion coefficient {k_name} seems too large (|{k_name}| > 1.0)"

        # Validate tangential distortion coefficients if enabled
        if params.get('use_tangential', False):
            for p_name, p_val in [('p1', params['p1']), ('p2', params['p2'])]:
                if abs(p_val) > 0.1:
                    return False, f"Tangential distortion coefficient {p_name} seems too large (|{p_name}| > 0.1)"

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
        print('accept: self.camera is', self.camera)
        print('accept: self.camera.camera_enu is', self.camera.camera_enu if self.camera is not None else None)
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
            from sudrabainiemakoni.cloudimage_camera import Camera, camera_from_dict

            params = self.get_modified_parameters()
            image_size = (cloudImage.imagearray.shape[1], cloudImage.imagearray.shape[0])

            # Determine distortion type name from checkbox state
            if params['use_rational']:
                distortion_type = 'rationaldistortionlimited'
            else:
                distortion_type = 'brownlensdistortionlimited'

            # Build a ct camera dict using raw tilt/heading/roll so camera_from_dict
            # sets all parameters correctly (elevation → tilt conversion done here)
            tilt = params['elevation'] + 90
            variables = {
                'focallength_x_px': params['fx'],
                'focallength_y_px': params['fy'],
                'center_x_px': params['cx'],
                'center_y_px': params['cy'],
                'image_width_px': image_size[0],
                'image_height_px': image_size[1],
                'heading_deg': params['azimuth'],
                'tilt_deg': tilt,
                'roll_deg': params['rotation'],
                'pos_x_m': 0.0,
                'pos_y_m': 0.0,
                'elevation_m': 0.0,
                'k1': params['k1'],
                'k2': params['k2'],
                'k3': params['k3'],
                'k4': params['k4'],
                'k5': params['k5'],
                'k6': params['k6'],
                'p1': params['p1'],
                'p2': params['p2'],
                'distortiontype': distortion_type,
                'projectiontype': params['projection'],
            }

            camera = Camera(image_size)
            camera.camera_enu = camera_from_dict(variables)
            camera.camera_ecef = camera.camera_ecef_from_camera_enu(camera.camera_enu, cloudImage.location)

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