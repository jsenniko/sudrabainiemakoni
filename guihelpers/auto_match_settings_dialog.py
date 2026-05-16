"""
Auto Star Match Settings Dialog - UI for configuring automatic star matching parameters.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QDoubleSpinBox, QSpinBox, QCheckBox,
                             QPushButton, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt


class AutoMatchSettingsDialog(QDialog):
    """Dialog for configuring automatic star matching parameters."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Star Matching Settings")
        self.setModal(True)

        from guihelpers.settings import AutoMatchSettings
        self._settings = settings if settings is not None else AutoMatchSettings()
        self._setup_ui()

    def _setup_ui(self):
        s = self._settings
        layout = QVBoxLayout()

        # Coarse matching group
        coarse_group = QGroupBox("Coarse Matching (triangle angular distances)")
        coarse_layout = QFormLayout()

        self.max_mag_coarse_spin = QDoubleSpinBox()
        self.max_mag_coarse_spin.setRange(0.0, 8.0)
        self.max_mag_coarse_spin.setSingleStep(0.5)
        self.max_mag_coarse_spin.setDecimals(1)
        self.max_mag_coarse_spin.setValue(s.max_magnitude_coarse)
        coarse_layout.addRow("Max magnitude (coarse):", self.max_mag_coarse_spin)

        self.angle_tol_spin = QDoubleSpinBox()
        self.angle_tol_spin.setRange(0.01, 2.0)
        self.angle_tol_spin.setSingleStep(0.01)
        self.angle_tol_spin.setDecimals(3)
        self.angle_tol_spin.setSuffix(" deg")
        self.angle_tol_spin.setValue(s.angle_tol_deg)
        coarse_layout.addRow("Angular tolerance:", self.angle_tol_spin)

        self.fl_uncertainty_spin = QDoubleSpinBox()
        self.fl_uncertainty_spin.setRange(0.0, 0.1)
        self.fl_uncertainty_spin.setSingleStep(0.001)
        self.fl_uncertainty_spin.setDecimals(4)
        self.fl_uncertainty_spin.setValue(s.focal_length_uncertainty)
        coarse_layout.addRow("Focal length uncertainty:", self.fl_uncertainty_spin)

        self.n_search_spin = QSpinBox()
        self.n_search_spin.setRange(10, 200)
        self.n_search_spin.setSingleStep(10)
        self.n_search_spin.setValue(s.n_search)
        coarse_layout.addRow("Stars used in search:", self.n_search_spin)

        coarse_group.setLayout(coarse_layout)
        layout.addWidget(coarse_group)

        # Fine matching group
        fine_group = QGroupBox("Fine Matching (nearest neighbour)")
        fine_layout = QFormLayout()

        self.max_mag_fine_spin = QDoubleSpinBox()
        self.max_mag_fine_spin.setRange(0.0, 8.0)
        self.max_mag_fine_spin.setSingleStep(0.5)
        self.max_mag_fine_spin.setDecimals(1)
        self.max_mag_fine_spin.setValue(s.max_magnitude_fine)
        fine_layout.addRow("Max magnitude (fine):", self.max_mag_fine_spin)

        self.nn_dist_spin = QDoubleSpinBox()
        self.nn_dist_spin.setRange(1.0, 100.0)
        self.nn_dist_spin.setSingleStep(1.0)
        self.nn_dist_spin.setDecimals(1)
        self.nn_dist_spin.setSuffix(" px")
        self.nn_dist_spin.setValue(s.nn_max_dist_px)
        fine_layout.addRow("Max NN distance:", self.nn_dist_spin)

        fine_group.setLayout(fine_layout)
        layout.addWidget(fine_group)

        # Optimization group
        opt_group = QGroupBox("Optimization")
        opt_layout = QFormLayout()

        self.optimize_intrinsics_check = QCheckBox()
        self.optimize_intrinsics_check.setChecked(s.optimize_intrinsics)
        opt_layout.addRow("Optimize focal length and center:", self.optimize_intrinsics_check)

        self.update_camera_check = QCheckBox()
        self.update_camera_check.setChecked(s.update_camera)
        opt_layout.addRow("Update camera after matching:", self.update_camera_check)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # Residual thresholds group
        rms_group = QGroupBox("Residual Thresholds")
        rms_layout = QFormLayout()

        self.max_rms_coarse_spin = QDoubleSpinBox()
        self.max_rms_coarse_spin.setRange(0.5, 50.0)
        self.max_rms_coarse_spin.setSingleStep(0.5)
        self.max_rms_coarse_spin.setDecimals(1)
        self.max_rms_coarse_spin.setSuffix(" px")
        self.max_rms_coarse_spin.setValue(s.max_rms_coarse_px)
        rms_layout.addRow("Max RMS coarse:", self.max_rms_coarse_spin)

        self.max_rms_fine_spin = QDoubleSpinBox()
        self.max_rms_fine_spin.setRange(0.5, 50.0)
        self.max_rms_fine_spin.setSingleStep(0.5)
        self.max_rms_fine_spin.setDecimals(1)
        self.max_rms_fine_spin.setSuffix(" px")
        self.max_rms_fine_spin.setValue(s.max_rms_fine_px)
        rms_layout.addRow("Max RMS fine:", self.max_rms_fine_spin)

        rms_group.setLayout(rms_layout)
        layout.addWidget(rms_group)

        # Help text
        help_label = QLabel(
            "Angular tolerance: base tolerance for triangle side matching\n"
            "Focal length uncertainty: fractional error (0.003 = 0.3%)\n"
            "Stars in search: brightest N stars used for coarse matching"
        )
        help_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(help_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_settings(self):
        from guihelpers.settings import AutoMatchSettings
        return AutoMatchSettings(
            max_magnitude_coarse     = self.max_mag_coarse_spin.value(),
            angle_tol_deg            = self.angle_tol_spin.value(),
            focal_length_uncertainty = self.fl_uncertainty_spin.value(),
            n_search                 = self.n_search_spin.value(),
            max_magnitude_fine       = self.max_mag_fine_spin.value(),
            nn_max_dist_px           = self.nn_dist_spin.value(),
            optimize_intrinsics      = self.optimize_intrinsics_check.isChecked(),
            max_rms_coarse_px        = self.max_rms_coarse_spin.value(),
            max_rms_fine_px          = self.max_rms_fine_spin.value(),
            update_camera            = self.update_camera_check.isChecked(),
        )


def show_auto_match_settings_dialog(parent=None, settings=None):
    """
    Show the auto star match settings dialog.

    Returns:
        (accepted, AutoMatchSettings)
    """
    from guihelpers.settings import AutoMatchSettings
    if settings is None:
        settings = AutoMatchSettings()

    dialog = AutoMatchSettingsDialog(parent=parent, settings=settings)
    accepted = dialog.exec_() == QDialog.Accepted
    if accepted:
        return True, dialog.get_settings()
    return False, settings
