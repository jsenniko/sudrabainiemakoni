from abc import ABC, abstractmethod

class Camera(ABC):
	def __init__(self):
		pass
		
	@abstractmethod
	def imageFromSpace(self, xyz, hide_backpoints=False):
		pass
		
		

# wrapper for cametransform.camera
class Cameratransform(Camera):
	def __init__(self, camera):
		self.camera  = camera
	def imageFromSpace(self, xyz, hide_backpoints=False):
		return self.camera.imageFromSpace(xyz, hide_backpoints=hide_backpoints)