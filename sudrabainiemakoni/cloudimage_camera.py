"""
Refactored Camera class extracted from cloudimage.py
This is the first step of a larger refactoring effort to modernize the camera system.

Changes made in this version:
- Removed obsolete _fit_metropolis method
- Simplified Fit method to only use 'optnew' optimization
- Removed unused scipy.optimize import
- Cleaned up method parameters

Future refactoring will provide:
- Better separation of concerns
- Enhanced parameter management
- Improved error handling
- More flexible architecture similar to cameratransform
"""

import os
import numpy as np
import cameratransform as ct
import pymap3d
from scipy.spatial.transform import Rotation
from sudrabainiemakoni import utils
from sudrabainiemakoni import optimize_camera
import cameraprojections
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class DistortionOrder(IntEnum):
    """Enumeration for camera distortion correction orders"""
    NONE = 0
    FIRST_ORDER = 1  
    SECOND_ORDER = 2
    THIRD_ORDER = 3

ProjectionType = Literal['rectilinear', 'equirectangular', 'stereographic']

@dataclass
class CameraCalibrationParams:
    """
    Parameters for camera calibration fitting process.
    
    Attributes:
        distortion: Order of lens distortion correction (0-3)
        centers: Whether to optimize camera center position
        separate_x_y: Whether to use separate focal lengths for X and Y axes
        projectiontype: Type of camera projection model
    """
    distortion: DistortionOrder = DistortionOrder.NONE
    centers: bool = True
    separate_x_y: bool = True
    projectiontype: ProjectionType = 'rectilinear'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for **kwargs unpacking to Camera.Fit()"""
        return {
            'distortion': int(self.distortion),
            'centers': self.centers,
            'separate_x_y': self.separate_x_y,
            'projectiontype': self.projectiontype
        }
    
    @classmethod
    def from_dict(cls, params: dict) -> 'CameraCalibrationParams':
        """Create CameraCalibrationParams from dictionary"""
        return cls(
            distortion=DistortionOrder(params.get('distortion', 0)),
            centers=params.get('centers', True),
            separate_x_y=params.get('separate_x_y', True),
            projectiontype=params.get('projectiontype', 'rectilinear')
        )
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate parameter values.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(self.distortion, (int, DistortionOrder)) or not (0 <= int(self.distortion) <= 3):
            return False, f"Distortion order must be 0-3, got {self.distortion}"
        
        valid_projections = ['rectilinear', 'equirectangular', 'stereographic']
        if self.projectiontype not in valid_projections:
            return False, f"Projection type must be one of {valid_projections}, got {self.projectiontype}"
        
        if not isinstance(self.centers, bool):
            return False, f"Centers parameter must be boolean, got {type(self.centers)}"
        
        if not isinstance(self.separate_x_y, bool):
            return False, f"Separate X/Y parameter must be boolean, got {type(self.separate_x_y)}"
        
        return True, ""
    
    def get_distortion_description(self) -> str:
        """Get human-readable description of distortion order"""
        descriptions = {
            0: "None (No distortion correction)",
            1: "First Order (Linear distortion)",
            2: "Second Order (Quadratic distortion)", 
            3: "Third Order (Cubic distortion)"
        }
        return descriptions.get(int(self.distortion), "Unknown")
    
    def get_projection_description(self) -> str:
        """Get human-readable description of projection type"""
        descriptions = {
            'rectilinear': "Rectilinear (Standard perspective)",
            'equirectangular': "Equirectangular (360° panoramic)",
            'stereographic': "Stereographic (Wide-angle fisheye)"
        }
        return descriptions.get(self.projectiontype, "Unknown")


class Camera:
    def __init__(self, cloudImage):
        self.cloudImage = cloudImage
        self.camera_enu = None
        self.camera_ecef = None
        
    def __getstate__(self):
        state = {}
        for ckey, cam in [('ENU',self.camera_enu), ('ECEF',self.camera_ecef)]:
            if cam is not None:
                keys = cam.parameters.parameters.keys()
                state[ckey] = {key: getattr(cam, key) for key in keys}
                state[ckey]['projectiontype'] =  cameraprojections.name_by_projection(cam.projection)
        return state
        
    def __setstate__(self, state):
        if 'camera_enu' in state:
            # fallback to default pickle saved with previous versions
            self.__dict__ = state
        else:
            self.camera_enu = None
            self.camera_ecef = None
            for ckey in ['ENU','ECEF']:
                if ckey in state:
                    if 'projectiontype' in state[ckey]:
                        projection = cameraprojections.projection_by_name(state[ckey]['projectiontype'])
                    else:
                        projection = ct.RectilinearProjection
                    cam = ct.Camera(projection(),  
                          ct.SpatialOrientation(),
                          ct.BrownLensDistortion())
                    for key in state[ckey]:
                        setattr(cam, key, state[ckey][key])
                    if ckey=='ENU':
                        self.camera_enu=cam
                    else:
                        self.camera_ecef=cam

    def residual(self, x, camera, space_coords, pxls):
        camera.heading_deg=x[0]
        camera.tilt_deg=x[1]
        camera.roll_deg=x[2]
        camera.focallength_x_px=x[3]
        camera.focallength_y_px=x[4]
        camera.center_x_px=x[5]
        camera.center_y_px=x[6]
        camera.k1=x[7]
        return np.sqrt(np.mean((camera.imageFromSpace(space_coords)-pxls)**2))

    def Fit(self, distortion=0, centers=True, separate_x_y=True, projectiontype='rectilinear'):
        sensor_size = (36, 24)
        focallength = utils.getExifEquivalentFocalLength35mm(self.cloudImage.filename)

        # if focal length given in exif specify range f/2 - 2*f
        f_bounds = np.array([5, 600]) if (focallength is None or focallength<=0.0) else np.array([focallength/2.0, focallength*2.0])
        f_bounds = f_bounds/sensor_size[0]*self.cloudImage.imagearray.shape[1]

        focallength = 24 if focallength is None else focallength
        focallength_px=focallength/sensor_size[0]*self.cloudImage.imagearray.shape[1]

        image_size = (self.cloudImage.imagearray.shape[1],  self.cloudImage.imagearray.shape[0])

        cx_bounds = [0, self.cloudImage.imagearray.shape[1]]
        cy_bounds = [0, self.cloudImage.imagearray.shape[0]]
        projection=cameraprojections.projection_by_name(projectiontype)
        print('Projection type:', projection)

        self.camera_enu = ct.Camera(projection(focallength_mm=focallength,   
                                         sensor=sensor_size,
                                         image=image_size),
                  ct.SpatialOrientation(elevation_m=0),
                  ct.BrownLensDistortion())

        enu_unit_coords = self.cloudImage.get_stars_enu_unit_coords()
        pxls = self.cloudImage.getPixelCoords()
        self.camera_enu.addLandmarkInformation(pxls, enu_unit_coords, [0.01, 0.01, 0.01])
        self.camera_enu.pos_x_m=0
        self.camera_enu.pos_y_m=0
        self.camera_enu.elevation_m=0
        self.camera_enu = optimize_camera.OptimizeCamera(self.camera_enu, enu_unit_coords, pxls, distortion=distortion, centers=centers,separate_x_y=separate_x_y,
                                                         f_bounds=f_bounds, cx_bounds=cx_bounds, cy_bounds=cy_bounds)

        calculated_star_px_coords = self.camera_enu.imageFromSpace(enu_unit_coords)
        residuals = calculated_star_px_coords - pxls
        print('ENU camera res:',np.sqrt(np.mean(residuals**2)))
        for sr, delta in zip(self.cloudImage.starReferences, residuals):
            print(sr, np.abs(delta))

        self.camera_ecef =  self.camera_ecef_from_camera_enu(self.camera_enu)

        ecef_unit = pymap3d.enu2ecef(enu_unit_coords[:,0], enu_unit_coords[:,1], enu_unit_coords[:,2],
                                      self.cloudImage.location.lat.value,self.cloudImage.location.lon.value,self.cloudImage.location.height.value)
        ecef_unit =np.array(ecef_unit).T

        print('ECEF camera res:',np.sqrt(np.mean((self.camera_ecef.imageFromSpace(ecef_unit)-pxls)**2)))

        return

    def camera_ecef_from_camera_enu(self, camera_enu):
        import copy
        # ja lieto astropy objektus, tad pārveidot no radiāniem uz grādiem nevajag
        sinlon, coslon = np.sin(self.cloudImage.location.lon).value,np.cos(self.cloudImage.location.lon).value
        sinlat, coslat = np.sin(self.cloudImage.location.lat).value,np.cos(self.cloudImage.location.lat).value
        rotMatr_uvw_enu=np.array([
            [-sinlon, coslon, 0],
            [-sinlat*coslon, -sinlat*sinlon, coslat],
            [coslat*coslon, coslat*sinlon, sinlat]
        ])
        angl = optimize_camera.Orientation_fromRotation(optimize_camera.Rotation_fromOrientation(camera_enu)*Rotation.from_matrix(rotMatr_uvw_enu))

        camera_ecef = ct.Camera(copy.deepcopy(camera_enu.projection),
                                ct.SpatialOrientation(**angl),
                                copy.deepcopy(camera_enu.lens))
        camera_ecef.pos_x_m=self.cloudImage.location.x.value
        camera_ecef.pos_y_m=self.cloudImage.location.y.value
        camera_ecef.elevation_m=self.cloudImage.location.z.value
        return camera_ecef

    def save(self, filename):
        optimize_camera.save_camera(self.camera_enu,os.path.splitext(filename)[0]+'_enu.json')
        optimize_camera.save_camera(self.camera_ecef,os.path.splitext(filename)[0]+'_ecef.json')
        
    def load(self, filename):
        fn = os.path.splitext(filename)[0]
        if os.path.exists(fn+'_enu.json'):
            fn=fn+'_enu.json'
        else:
            fn = fn.rsplit('_',1)[0]+'_enu.json'
        if os.path.exists(fn):
            self.camera_enu = optimize_camera.load_camera(fn)
        fn = os.path.splitext(filename)[0]
        if os.path.exists(fn+'_ecef.json'):
            fn=fn+'_ecef.json'
        else:
            fn = fn.rsplit('_',1)[0]+'_ecef.json'
        if os.path.exists(fn):
            self.camera_ecef = optimize_camera.load_camera(fn)

    def imageFromECEF(self, xyz):
        return self.camera_ecef.imageFromSpace(xyz)
        
    def imageFromAltAz(self, az,alt):
        enu = pymap3d.aer2enu(az,alt,1.0)
        return self.camera_enu.imageFromSpace(enu)

    def get_azimuth_elevation_rotation(self):
        camera_enu=self.camera_enu
        if camera_enu.tilt_deg<0:
            azimuth =180+camera_enu.heading_deg if camera_enu.heading_deg<0 else camera_enu.heading_deg-180
            tilt =-camera_enu.tilt_deg
            roll =camera_enu.roll_deg-180 if camera_enu.roll_deg>0 else 180+camera_enu.roll_deg
        else:
            azimuth =camera_enu.heading_deg
            tilt =camera_enu.tilt_deg
            roll =camera_enu.roll_deg
        # tilt - 0 deg down, zenith distance=180-tilt, elevation=tilt-90
        return azimuth, tilt-90, roll
        
    def get_focal_lengths_mm(self):
        camera_enu=self.camera_enu
        return camera_enu.focallength_x_px * 36.0 / camera_enu.image_width_px, camera_enu.focallength_y_px * 36.0 / camera_enu.image_width_px, camera_enu.center_x_px, camera_enu.center_y_px
        
    @classmethod
    def ecef_from_enu(cls, camera_enu, lat_deg, lon_deg, height_m):
        import copy
        # ja lieto astropy objektus, tad pārveidot no radiāniem uz grādiem nevajag
        lon, lat = np.radians(lon_deg), np.radians(lat_deg)
        sinlon, coslon = np.sin(lon),np.cos(lon)
        sinlat, coslat = np.sin(lat),np.cos(lat)
        rotMatr_uvw_enu=np.array([
            [-sinlon, coslon, 0],
            [-sinlat*coslon, -sinlat*sinlon, coslat],
            [coslat*coslon, coslat*sinlon, sinlat]
        ])
        angl = optimize_camera.Orientation_fromRotation(optimize_camera.Rotation_fromOrientation(camera_enu)*Rotation.from_matrix(rotMatr_uvw_enu))

        camera_ecef = ct.Camera(copy.deepcopy(camera_enu.projection),
                                ct.SpatialOrientation(**angl),
                                copy.deepcopy(camera_enu.lens))
        x, y, z = pymap3d.geodetic2ecef(lat_deg, lon_deg, height_m)
        camera_ecef.pos_x_m=x
        camera_ecef.pos_y_m=y
        camera_ecef.elevation_m=z
        return camera_ecef
        
    @classmethod
    def make_cameras(cls, lat_deg, lon_deg, height_m, width_px, height_px, focallength_35mm, azimuth, elevation, rotation, projectiontype='rectilinear'):
        projection=cameraprojections.projection_by_name(projectiontype)
        camera_enu = ct.Camera(projection(focallength_mm=focallength_35mm,  
                                             sensor=(36,24),
                                             image=(width_px, height_px)),
                               ct.SpatialOrientation(
                                   heading_deg = azimuth,
                                   tilt_deg = 90 + elevation,
                                   roll_deg = rotation,
                                   elevation_m=0.0
                               ),  ct.BrownLensDistortion())
        camera_enu.pos_x_m=0
        camera_enu.pos_y_m=0
        camera_enu.elevation_m=0

        camera_ecef=Camera.ecef_from_enu(camera_enu, lat_deg, lon_deg, height_m)
        return camera_enu, camera_ecef
        
    @classmethod
    def GetAltAzGrid_fromcamera(cls, width_px, height_px, camera_enu):
        i_grid, j_grid = np.meshgrid(np.arange(width_px),np.arange(height_px))
        grid_points=np.array([i_grid.flatten(), j_grid.flatten()]).T
        enu = camera_enu.spaceFromImage(grid_points, D=1)
        azalt = pymap3d.enu2aer(*enu.T)[0:2]
        azalt = np.reshape(np.array(azalt),(2,i_grid.shape[0],i_grid.shape[1]))
        az_min, az_max=azalt[0].min(), azalt[0].max()
        if az_max-az_min>180:
            azalt[0]=np.where(azalt[0]>180,azalt[0]-360,azalt[0])
        return azalt