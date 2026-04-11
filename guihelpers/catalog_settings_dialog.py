"""
Catalog Settings Dialog - UI for configuring catalog star display parameters.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                            QDoubleSpinBox, QPushButton, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt


class CatalogSettingsDialog(QDialog):
    """
    Dialog for configuring catalog star display settings.
    Allows user to set magnitude range and altitude filter.
    """

    def __init__(self, parent=None, min_magnitude=-3, max_magnitude=4,
                 min_altitude=0.0, overshoot_px=20):
        """
        Initialize the catalog settings dialog.

        Args:
            parent: Parent widget
            min_magnitude: Initial minimum magnitude (default: -3)
            max_magnitude: Initial maximum magnitude (default: 4)
            min_altitude: Initial minimum altitude in degrees (default: 0.0)
            overshoot_px: Initial overshoot in pixels (default: 20)
        """
        super().__init__(parent)
        self.setWindowTitle("Catalog Star Settings")
        self.setModal(True)

        # Store initial values
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.min_altitude = min_altitude
        self.overshoot_px = overshoot_px

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout()

        # Magnitude settings group
        mag_group = QGroupBox("Magnitude Range")
        mag_layout = QFormLayout()

        # Minimum magnitude
        self.min_mag_spin = QDoubleSpinBox()
        self.min_mag_spin.setRange(-5.0, 10.0)
        self.min_mag_spin.setSingleStep(0.5)
        self.min_mag_spin.setDecimals(1)
        self.min_mag_spin.setValue(self.min_magnitude)
        mag_layout.addRow("Minimum Magnitude:", self.min_mag_spin)

        # Maximum magnitude
        self.max_mag_spin = QDoubleSpinBox()
        self.max_mag_spin.setRange(-5.0, 10.0)
        self.max_mag_spin.setSingleStep(0.5)
        self.max_mag_spin.setDecimals(1)
        self.max_mag_spin.setValue(self.max_magnitude)
        mag_layout.addRow("Maximum Magnitude:", self.max_mag_spin)

        mag_group.setLayout(mag_layout)
        layout.addWidget(mag_group)

        # Altitude filter group
        alt_group = QGroupBox("Altitude Filter")
        alt_layout = QFormLayout()

        # Minimum altitude
        self.min_alt_spin = QDoubleSpinBox()
        self.min_alt_spin.setRange(-90.0, 90.0)
        self.min_alt_spin.setSingleStep(5.0)
        self.min_alt_spin.setDecimals(1)
        self.min_alt_spin.setSuffix(" °")
        self.min_alt_spin.setValue(self.min_altitude)
        alt_layout.addRow("Minimum Altitude:", self.min_alt_spin)

        alt_group.setLayout(alt_layout)
        layout.addWidget(alt_group)

        # Advanced settings group
        adv_group = QGroupBox("Advanced")
        adv_layout = QFormLayout()

        # Overshoot pixels
        self.overshoot_spin = QDoubleSpinBox()
        self.overshoot_spin.setRange(0, 100)
        self.overshoot_spin.setSingleStep(5)
        self.overshoot_spin.setDecimals(0)
        self.overshoot_spin.setSuffix(" px")
        self.overshoot_spin.setValue(self.overshoot_px)
        adv_layout.addRow("Edge Overshoot:", self.overshoot_spin)

        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)

        # Help text
        help_label = QLabel(
            "Magnitude: Lower values = brighter stars\n"
            "Altitude: Minimum angle above horizon\n"
            "Edge Overshoot: Include stars slightly outside image"
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

    def get_values(self):
        """
        Get the configured values from the dialog.

        Returns:
            tuple: (min_magnitude, max_magnitude, min_altitude, overshoot_px)
        """
        return (
            self.min_mag_spin.value(),
            self.max_mag_spin.value(),
            self.min_alt_spin.value(),
            self.overshoot_spin.value()
        )


def show_catalog_settings_dialog(parent=None, min_magnitude=-3, max_magnitude=4,
                                 min_altitude=0.0, overshoot_px=20):
    """
    Show the catalog settings dialog.

    Args:
        parent: Parent widget
        min_magnitude: Initial minimum magnitude
        max_magnitude: Initial maximum magnitude
        min_altitude: Initial minimum altitude
        overshoot_px: Initial overshoot pixels

    Returns:
        tuple: (accepted, (min_mag, max_mag, min_alt, overshoot))
               accepted is True if user clicked OK, False if cancelled
    """
    dialog = CatalogSettingsDialog(
        parent=parent,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        min_altitude=min_altitude,
        overshoot_px=overshoot_px
    )

    result = dialog.exec_()
    accepted = result == QDialog.Accepted

    if accepted:
        return accepted, dialog.get_values()
    else:
        return accepted, (min_magnitude, max_magnitude, min_altitude, overshoot_px)
