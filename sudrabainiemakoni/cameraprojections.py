import numpy as np
import cameratransform as ct
from cameratransform import CameraProjection
class StereographicProjection(CameraProjection):
	r"""
	ex. Samyang 8 mm

	**Projection**:

	"""

	def getRay(self, points, normed=False):
		# ensure that the points are provided as an array
		points = np.array(points)
		# set z=focallenth and solve the other equations for x and y
		ixr = -(points[...,0] - self.center_x_px) / self.focallength_x_px/2
		iyr = (points[...,1] - self.center_y_px) / self.focallength_y_px/2
		
		xr = 2*ixr/(1+ixr**2+iyr**2)
		yr = 2*iyr/(1+ixr**2+iyr**2)
		zr = (1-(ixr**2+iyr**2))/(1+ixr**2+iyr**2)
		
		
		ray = np.array([xr,
						yr,
						zr]).T
		# norm the ray if desired
		if normed:
			ray /= np.linalg.norm(ray, axis=-1)[..., None]
		# return the ray
		return -ray

	def imageFromCamera(self, points, hide_backpoints=True):
		"""
						 ( x )                                         (       y       )
			x_im = f_x * (---) + offset_x                y_im = f_y * (---------------) + offset_y
						 ( z-sqrt(x^2+y^2+z^2) )                       (z-sqrt(x**2+y**2+z**2))
		"""
		points = np.array(points)
		# set small z distances to 0
		points[np.abs(points[..., 2]) < 1e-10] = 0
		# transform the points
		L = np.sqrt(points[...,0]**2+points[...,1]**2+points[...,2]**2)
		transformed_points = np.array([-2*points[..., 0] * self.focallength_x_px / (points[..., 2] - L) + self.center_x_px,
									   2*points[..., 1]  * self.focallength_y_px /  (points[..., 2] - L) + self.center_y_px]).T
		if hide_backpoints:
			transformed_points[points[..., 2] > 0] = np.nan
		return transformed_points

class EquirectangularProjection(CameraProjection):
	r"""
	This projection is a common projection used for panoranic images. The projection can cover the
	full range of angles in both x and y direction.

	**Projection**:

	.. math::
		x_\mathrm{im} &= f_x \cdot \arctan{\left(\frac{x}{z}\right)} + c_x\\
		y_\mathrm{im} &= f_y \cdot \arctan{\left(\frac{y}{\sqrt{x^2+z^2}}\right)} + c_y

	**Rays**:

	.. math::
		\vec{r} = \begin{pmatrix}
			-\sin\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)\\
			\tan\left(\frac{y_\mathrm{im} - c_y}{f_y}\right)\\
			\cos\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)
		\end{pmatrix}
	"""

	def getRay(self, points, normed=False):
		# ensure that the points are provided as an array
		points = np.array(points)
		# set r=1 and solve the other equations for x and y
		r = 1
		alpha = (points[..., 0] - self.center_x_px) / self.focallength_x_px
		x = -np.sin(alpha) * r
		z = np.cos(alpha) * r
		y = r * np.tan((points[..., 1] - self.center_y_px) / self.focallength_y_px)
		# compose the ray
		ray = np.array([x, y, z]).T
		# norm the ray if desired
		if normed:
			ray /= np.linalg.norm(ray, axis=-1)[..., None]
		# return the rey
		return -ray

	def imageFromCamera(self, points, hide_backpoints=True):
		"""
							   ( x )                                    (       y       )
			x_im = f_x * arctan(---) + offset_x      y_im = f_y * arctan(---------------) + offset_y
							   ( z )                                    (sqrt(x**2+z**2))
		"""
		# ensure that the points are provided as an array
		points = np.array(points)
		# set small z distances to 0
		points[np.abs(points[..., 2]) < 1e-10] = 0
		# transform the points
		transformed_points = np.array(
			[-self.focallength_x_px * np.arctan2(-points[..., 0], -points[..., 2]) + self.center_x_px,
			 -self.focallength_y_px * np.arctan2(points[..., 1], np.sqrt(
				 points[..., 0] ** 2 + points[..., 2] ** 2)) + self.center_y_px]).T

		# return the points
		return transformed_points

	def getFieldOfView(self):
		return np.rad2deg(self.image_width_px / self.focallength_x_px), \
			   np.rad2deg(self.image_height_px / self.focallength_y_px)

	def focallengthFromFOV(self, view_x=None, view_y=None):
		if view_x is not None:
			return self.image_width_px / np.deg2rad(view_x)
		else:
			return self.image_height_px / np.deg2rad(view_y)

	def imageFromFOV(self, view_x=None, view_y=None):
		if view_x is not None:
			# image_width_mm
			return self.focallength_x_px * np.deg2rad(view_x)
		else:
			# image_height_mm
			return self.focallength_y_px * np.deg2rad(view_y)		
		
def projection_by_name(name):
	if name == 'equirectangular':
		projection = EquirectangularProjection
	elif name == 'stereographic':
		projection = StereographicProjection
	else:
		projection = ct.RectilinearProjection
	return projection
def name_by_projection(projection):
	if type(projection) == StereographicProjection:
		return 'stereographic'
	elif type(projection) == EquirectangularProjection:
		return 'equirectangular'
	else:
		return 'rectilinear'

