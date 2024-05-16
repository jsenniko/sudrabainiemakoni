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
		ixr = -(points[...,0] - self.center_x_px) / self.focallength_x_px
		iyr = (points[...,1] - self.center_y_px) / self.focallength_y_px
		print(ixr,iyr)
		
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
						 ( z+sqrt(x^2+y^2+z^2) )                       (z+sqrt(x**2+y**2+z**2))
		"""
		points = np.array(points)
		# set small z distances to 0
		points[np.abs(points[..., 2]) < 1e-10] = 0
		# transform the points
		L = np.sqrt(points[...,0]**2+points[...,1]**2+points[...,2]**2)
		#print(points)
		#print(self.focallength_x_px, self.focallength_y_px)
		#print(L)
		transformed_points = np.array([-points[..., 0] * self.focallength_x_px / (points[..., 2] - L) + self.center_x_px,
									   points[..., 1]  * self.focallength_y_px /  (points[..., 2] - L) + self.center_y_px]).T
		if hide_backpoints:
			transformed_points[points[..., 2] > 0] = np.nan
		return transformed_points

		
		
def projection_by_name(name):
	if name == 'equirectangular':
		projection = ct.EquirectangularProjection
	elif name == 'stereographic':
		projection = StereographicProjection
	else:
		projection = ct.RectilinearProjection
	return projection
def name_by_projection(projection):
	if type(projection) == StereographicProjection:
		return 'stereographic'
	elif type(projection) == ct.EquirectangularProjection:
		return 'equirectangular'
	else:
		return 'rectilinear'

