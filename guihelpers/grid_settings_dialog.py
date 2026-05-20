"""
Grid Settings Dialog - UI for configuring altitude/azimuth grid display parameters.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                            QDoubleSpinBox, QPushButton, QGroupBox, QFormLayout,
                            QComboBox, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt


class GridSettings:
    """Container for grid settings parameters."""
    def __init__(self):
        # Grid density
        self.minor_az_step = None
        self.major_az_step = None
        self.minor_alt_step = None
        self.major_alt_step = None
        self.nticks = 15
        self.nticks_major = 5

        # Grid styles
        self.minor_linewidth = 0.3
        self.major_linewidth = 0.8
        self.minor_linestyle = ':'
        self.major_linestyle = '--'
        self.grid_color = '#DDFFDD'

        # Label settings
        self.label_major_only = True
        self.label_placement = 'outside'
        self.label_side = 'all'
        self.label_pad = 10

    def to_grid_kwargs(self):
        """Convert settings to grid_kwargs dictionary for DrawAltAzGrid_v2."""
        grid_kwargs = {
            'nticks': self.nticks,
            'nticks_major': self.nticks_major,
            'label_major_only': self.label_major_only,
            'label_placement': self.label_placement,
            'label_side': self.label_side,
            'label_pad': self.label_pad,
        }

        # Add step sizes if specified
        if self.minor_az_step is not None:
            grid_kwargs['minor_az_step'] = self.minor_az_step
        if self.major_az_step is not None:
            grid_kwargs['major_az_step'] = self.major_az_step
        if self.minor_alt_step is not None:
            grid_kwargs['minor_alt_step'] = self.minor_alt_step
        if self.major_alt_step is not None:
            grid_kwargs['major_alt_step'] = self.major_alt_step

        # Add grid styles
        grid_kwargs['minor_az_style'] = {
            'color': self.grid_color,
            'linestyle': self.minor_linestyle,
            'linewidth': self.minor_linewidth
        }
        grid_kwargs['major_az_style'] = {
            'color': self.grid_color,
            'linestyle': self.major_linestyle,
            'linewidth': self.major_linewidth
        }
        grid_kwargs['minor_alt_style'] = {
            'color': self.grid_color,
            'linestyle': self.minor_linestyle,
            'linewidth': self.minor_linewidth
        }
        grid_kwargs['major_alt_style'] = {
            'color': self.grid_color,
            'linestyle': self.major_linestyle,
            'linewidth': self.major_linewidth
        }

        return grid_kwargs


class GridSettingsDialog(QDialog):
    """
    Dialog for configuring altitude/azimuth grid display settings.
    """

    def __init__(self, parent=None, settings=None):
        """
        Initialize the grid settings dialog.

        Args:
            parent: Parent widget
            settings: GridSettings object with initial values
        """
        super().__init__(parent)
        self.setWindowTitle("Grid Settings")
        self.setModal(True)

        # Store initial settings
        self.settings = settings if settings is not None else GridSettings()

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout()

        # Grid density group
        density_group = QGroupBox("Grid Density")
        density_layout = QFormLayout()

        # Automatic mode
        mode_layout = QHBoxLayout()
        self.auto_mode_check = QCheckBox("Automatic (based on image extent)")
        self.auto_mode_check.setChecked(self.settings.minor_az_step is None)
        self.auto_mode_check.stateChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.auto_mode_check)
        density_layout.addRow("Mode:", mode_layout)

        # Automatic settings
        self.nticks_spin = QSpinBox()
        self.nticks_spin.setRange(5, 50)
        self.nticks_spin.setValue(self.settings.nticks)
        self.nticks_spin.setSuffix(" ticks")
        density_layout.addRow("Minor Grid Ticks:", self.nticks_spin)

        self.nticks_major_spin = QSpinBox()
        self.nticks_major_spin.setRange(2, 20)
        self.nticks_major_spin.setValue(self.settings.nticks_major)
        self.nticks_major_spin.setSuffix(" ticks")
        density_layout.addRow("Major Grid Ticks:", self.nticks_major_spin)

        # Manual settings
        self.minor_az_spin = QDoubleSpinBox()
        self.minor_az_spin.setRange(0.1, 45.0)
        self.minor_az_spin.setSingleStep(1.0)
        self.minor_az_spin.setDecimals(1)
        self.minor_az_spin.setSuffix(" °")
        self.minor_az_spin.setValue(self.settings.minor_az_step if self.settings.minor_az_step is not None else 1.0)
        density_layout.addRow("Minor Azimuth Step:", self.minor_az_spin)

        self.major_az_spin = QDoubleSpinBox()
        self.major_az_spin.setRange(1.0, 90.0)
        self.major_az_spin.setSingleStep(5.0)
        self.major_az_spin.setDecimals(1)
        self.major_az_spin.setSuffix(" °")
        self.major_az_spin.setValue(self.settings.major_az_step if self.settings.major_az_step is not None else 10.0)
        density_layout.addRow("Major Azimuth Step:", self.major_az_spin)

        self.minor_alt_spin = QDoubleSpinBox()
        self.minor_alt_spin.setRange(0.1, 45.0)
        self.minor_alt_spin.setSingleStep(1.0)
        self.minor_alt_spin.setDecimals(1)
        self.minor_alt_spin.setSuffix(" °")
        self.minor_alt_spin.setValue(self.settings.minor_alt_step if self.settings.minor_alt_step is not None else 1.0)
        density_layout.addRow("Minor Altitude Step:", self.minor_alt_spin)

        self.major_alt_spin = QDoubleSpinBox()
        self.major_alt_spin.setRange(1.0, 90.0)
        self.major_alt_spin.setSingleStep(5.0)
        self.major_alt_spin.setDecimals(1)
        self.major_alt_spin.setSuffix(" °")
        self.major_alt_spin.setValue(self.settings.major_alt_step if self.settings.major_alt_step is not None else 10.0)
        density_layout.addRow("Major Altitude Step:", self.major_alt_spin)

        density_group.setLayout(density_layout)
        layout.addWidget(density_group)

        # Grid style group
        style_group = QGroupBox("Grid Style")
        style_layout = QFormLayout()

        # Line widths
        self.minor_linewidth_spin = QDoubleSpinBox()
        self.minor_linewidth_spin.setRange(0.1, 5.0)
        self.minor_linewidth_spin.setSingleStep(0.1)
        self.minor_linewidth_spin.setDecimals(1)
        self.minor_linewidth_spin.setValue(self.settings.minor_linewidth)
        style_layout.addRow("Minor Line Width:", self.minor_linewidth_spin)

        self.major_linewidth_spin = QDoubleSpinBox()
        self.major_linewidth_spin.setRange(0.1, 5.0)
        self.major_linewidth_spin.setSingleStep(0.1)
        self.major_linewidth_spin.setDecimals(1)
        self.major_linewidth_spin.setValue(self.settings.major_linewidth)
        style_layout.addRow("Major Line Width:", self.major_linewidth_spin)

        # Line styles
        self.minor_linestyle_combo = QComboBox()
        self.minor_linestyle_combo.addItems(['-', '--', ':', '-.'])
        self.minor_linestyle_combo.setCurrentText(self.settings.minor_linestyle)
        style_layout.addRow("Minor Line Style:", self.minor_linestyle_combo)

        self.major_linestyle_combo = QComboBox()
        self.major_linestyle_combo.addItems(['-', '--', ':', '-.'])
        self.major_linestyle_combo.setCurrentText(self.settings.major_linestyle)
        style_layout.addRow("Major Line Style:", self.major_linestyle_combo)

        style_group.setLayout(style_layout)
        layout.addWidget(style_group)

        # Label settings group
        label_group = QGroupBox("Label Settings")
        label_layout = QFormLayout()

        # Label major only
        self.label_major_only_check = QCheckBox("Label major gridlines only")
        self.label_major_only_check.setChecked(self.settings.label_major_only)
        label_layout.addRow(self.label_major_only_check)

        # Label placement
        self.label_placement_combo = QComboBox()
        self.label_placement_combo.addItems(['outside', 'inside'])
        self.label_placement_combo.setCurrentText(self.settings.label_placement)
        label_layout.addRow("Label Placement:", self.label_placement_combo)

        # Label sides
        self.label_side_combo = QComboBox()
        self.label_side_combo.addItems(['all', 'left', 'right', 'top', 'bottom'])
        self.label_side_combo.setCurrentText(self.settings.label_side)
        label_layout.addRow("Label Sides:", self.label_side_combo)

        # Label padding
        self.label_pad_spin = QDoubleSpinBox()
        self.label_pad_spin.setRange(1, 100)
        self.label_pad_spin.setSingleStep(5)
        self.label_pad_spin.setDecimals(0)
        self.label_pad_spin.setSuffix(" px")
        self.label_pad_spin.setValue(self.settings.label_pad)
        label_layout.addRow("Label Padding:", self.label_pad_spin)

        label_group.setLayout(label_layout)
        layout.addWidget(label_group)

        # Help text
        help_label = QLabel(
            "<i>Automatic mode: Grid lines placed automatically based on image extent.<br>"
            "Manual mode: Specify exact degree intervals for grid lines.</i>"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Update UI state based on mode
        self._on_mode_changed()

    def _on_mode_changed(self):
        """Handle mode change between automatic and manual."""
        is_auto = self.auto_mode_check.isChecked()

        # Enable/disable controls based on mode
        self.nticks_spin.setEnabled(is_auto)
        self.nticks_major_spin.setEnabled(is_auto)

        self.minor_az_spin.setEnabled(not is_auto)
        self.major_az_spin.setEnabled(not is_auto)
        self.minor_alt_spin.setEnabled(not is_auto)
        self.major_alt_spin.setEnabled(not is_auto)

    def get_settings(self):
        """
        Get the current settings from the dialog.

        Returns:
            GridSettings object with current values
        """
        settings = GridSettings()

        # Get mode
        is_auto = self.auto_mode_check.isChecked()

        if is_auto:
            settings.minor_az_step = None
            settings.major_az_step = None
            settings.minor_alt_step = None
            settings.major_alt_step = None
            settings.nticks = self.nticks_spin.value()
            settings.nticks_major = self.nticks_major_spin.value()
        else:
            settings.minor_az_step = self.minor_az_spin.value()
            settings.major_az_step = self.major_az_spin.value()
            settings.minor_alt_step = self.minor_alt_spin.value()
            settings.major_alt_step = self.major_alt_spin.value()
            settings.nticks = self.nticks_spin.value()
            settings.nticks_major = self.nticks_major_spin.value()

        # Get style settings
        settings.minor_linewidth = self.minor_linewidth_spin.value()
        settings.major_linewidth = self.major_linewidth_spin.value()
        settings.minor_linestyle = self.minor_linestyle_combo.currentText()
        settings.major_linestyle = self.major_linestyle_combo.currentText()

        # Get label settings
        settings.label_major_only = self.label_major_only_check.isChecked()
        settings.label_placement = self.label_placement_combo.currentText()
        settings.label_side = self.label_side_combo.currentText()
        settings.label_pad = self.label_pad_spin.value()

        return settings


def show_grid_settings_dialog(parent=None, settings=None):
    """
    Show the grid settings dialog.

    Args:
        parent: Parent widget
        settings: GridSettings object with initial values

    Returns:
        Tuple of (accepted, settings) where accepted is bool and settings is GridSettings
    """
    dialog = GridSettingsDialog(parent, settings)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return True, dialog.get_settings()
    else:
        return False, settings
