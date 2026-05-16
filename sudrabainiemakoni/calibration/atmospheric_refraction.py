import numpy as np


def apply_atmospheric_refraction_correction(alt_deg):
    """
    Apply atmospheric refraction correction to altitude angle.

    Accounts for the bending of light through Earth's atmosphere, which causes
    celestial objects to appear slightly higher than their true position.
    Uses Bennett's empirical refraction formula.

    Args:
        alt_deg: Altitude angle(s) in degrees (scalar or array)

    Returns:
        Corrected altitude angle(s) in degrees
    """
    return alt_deg + 0.01666 / np.tan(np.radians(alt_deg + (7.31 / (alt_deg + 4.4))))
