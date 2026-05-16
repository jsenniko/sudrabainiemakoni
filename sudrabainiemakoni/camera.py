from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ProjectionParameters:
	focal_length_x: float
	focal_length_y: float
	center_x: float
	center_y: float
	image_width: int
	image_height: int

	def to_dict(self):
		return asdict(self)

	@classmethod
	def from_dict(cls, data):
		return cls(**data)


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