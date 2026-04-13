"""
FlexibleDoubleSpinBox - A QDoubleSpinBox that accepts pasted values with any precision.

The standard QDoubleSpinBox rejects pasted values if they have more decimal places
than configured. This custom widget accepts the value and rounds it appropriately.
"""

from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """
    Enhanced QDoubleSpinBox that accepts pasted values with any number of decimals.

    When a value is pasted (Ctrl+V), it will:
    1. Try to parse it as a float
    2. Round it to the configured decimal places
    3. Clamp it to min/max range
    4. Set the value

    This avoids the frustrating behavior where pasting "1.234567" into a field
    configured for 3 decimals would be rejected entirely.
    """

    def keyPressEvent(self, event: QKeyEvent):
        """Override key press to handle paste operations."""
        # Check for Ctrl+V (paste)
        if event.matches(QKeyEvent.Paste):
            self._handle_paste()
            event.accept()
            return

        # Default behavior for all other keys
        super().keyPressEvent(event)

    def _handle_paste(self):
        """Handle paste operation with flexible decimal precision."""
        from PyQt5.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()

        if not text:
            return

        # Try to parse as float
        try:
            value = float(text)

            # Clamp to min/max range
            value = max(self.minimum(), min(self.maximum(), value))

            # Set the value (will automatically round to decimals())
            self.setValue(value)

        except ValueError:
            # If it's not a valid number, let Qt handle it (will reject)
            pass
