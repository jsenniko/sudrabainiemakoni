import numpy as np
import cameratransform
from cameratransform.parameter_set import  ParameterSet, Parameter, TYPE_DISTORTION

class BrownLensDistortionLimited(cameratransform.BrownLensDistortion):
    r"""
    A modified version of Brown's distortion model that limits the valid radius range to ensure monotonicity.

    This class extends cameratransform.BrownLensDistortion by restricting the distortion to the radius range
    where the distortion function is monotonically increasing. For certain combinations of distortion
    coefficients (k1, k2, k3), the Brown distortion model can produce non-monotonic behavior at large radii,
    where the distorted radius starts to decrease. This implementation automatically detects and limits
    the valid radius to prevent such behavior.

    The distortion model follows the same mathematical formulation as the standard Brown model:

    Adjust scale and offset of x and y to be relative to the center:

    .. math::
        x' &= \frac{x-c_x}{f_x}\\
        y' &= \frac{y-c_y}{f_y}

    Transform the radius from the center with the distortion:

    .. math::
        r &= \sqrt{x'^2 + y'^2}\\
        r' &= r \cdot (1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6)\\
        x_\mathrm{distorted}' &= x' / r \cdot r'\\
        y_\mathrm{distorted}' &= y' / r \cdot r'

    Readjust scale and offset to obtain again pixel coordinates:

    .. math::
        x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot f_x + c_x\\
        y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot f_y + c_y

    The key difference from the base BrownLensDistortion is that points beyond the maximum monotonic
    radius (rmax) will produce NaN values in distortedFromImage, indicating invalid distortion regions.

    Attributes:
        _rmax: Maximum radius where the distortion function remains monotonically increasing.
    """
    def __init__(self, k1=None, k2=None, k3=None, projection=None):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            k1=Parameter(k1, default=0, range=(0, None), type=TYPE_DISTORTION),
            k2=Parameter(k2, default=0, range=(0, None), type=TYPE_DISTORTION),
            k3=Parameter(k3, default=0, range=(0, None), type=TYPE_DISTORTION),
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def _init_inverse(self):
        # increasing radius only! # can use also np.sqrt(np.roots([7*k3, 5*k2, 3*k1, 1])), but then hard to check: should check for existence of lowest real root etc.!
        r = np.arange(0, 2, 0.01)
        r_transformed = self._convert_radius_orig(r)
        invalid = np.diff(r_transformed) < 0.0

        if np.any(invalid):
            # Non-monotonic behavior detected, limit to valid range
            valid_to = np.argmax(invalid)
            r = r[:valid_to+1]
            r_transformed = r_transformed[:valid_to+1]
        # else: entire range is monotonic, use all points

        from scipy import interpolate
        self._rmax = r[-1]
        self._convert_radius = interpolate.InterpolatedUnivariateSpline(r, r_transformed, k=3, ext=3)
        inv_inter = interpolate.InterpolatedUnivariateSpline(r_transformed, r, k=3, ext=3)

        self._convert_radius_inverse = inv_inter
        if self.projection is not None:
            self.scale = np.array([self.projection.focallength_x_px, self.projection.focallength_y_px])
            self.offset = np.array([self.projection.center_x_px, self.projection.center_y_px])

    def _convert_radius_orig(self, r):
        return r*(1 + self.parameters.k1*r**2 + self.parameters.k2*r**4 + self.parameters.k3*r**6)

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # set nans to 0
        points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        r_transformed = self._convert_radius(r)
        r_transformed[r>self._rmax] = np.nan
        points = points / r * r_transformed
        # set nans to 0
        # points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset
    

from sudrabainiemakoni.cv2_lens_distortion import OpenCVBrownLensDistortion 

def distortion_by_name(name):
	if name == 'opencvbrownlensdistortion':
		distortion = OpenCVBrownLensDistortion
	else:
		distortion = BrownLensDistortionLimited
	return distortion
def name_by_distortion(distortion):
	return distortion.__class__.__name__.lower()
