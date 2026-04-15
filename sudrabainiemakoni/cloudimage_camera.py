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
from sudrabainiemakoni import optimize_camera
from sudrabainiemakoni import cameraprojections
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal
from sudrabainiemakoni.lensdistortions import distortion_by_name, name_by_distortion

class DistortionOrder(IntEnum):
    """Enumeration for camera distortion correction orders"""
    NONE = 0
    FIRST_ORDER = 1
    SECOND_ORDER = 2
    THIRD_ORDER = 3
    FOURTH_ORDER = 4  # Includes tangential distortion (p1, p2)

ProjectionType = Literal['rectilinear', 'equirectangular', 'stereographic']

@dataclass
class CameraCalibrationParams:
    """
    Parameters for camera calibration fitting process.

    Attributes:
        distortion: Order of lens distortion correction (0-4)
                   0=none, 1=k1, 2=k1+k2, 3=k1+k2+k3, 4=k1+k2+k3+p1+p2
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
        if not isinstance(self.distortion, (int, DistortionOrder)) or not (0 <= int(self.distortion) <= 4):
            return False, f"Distortion order must be 0-4, got {self.distortion}"
        
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
            1: "First Order (k1 only)",
            2: "Second Order (k1, k2)",
            3: "Third Order (k1, k2, k3)",
            4: "Fourth Order (k1, k2, k3, p1, p2 - includes tangential)"
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
    def __init__(self, image_size):
        """
        Initialize Camera.

        Args:
            image_size: Tuple of (width_px, height_px)
        """
        self.image_size = image_size
        self.camera_enu = None
        self.camera_ecef = None
    
    @classmethod
    def from_manual_parameters(cls, image_size, location, fx, fy, cx, cy, azimuth, 
                               elevation, rotation, k1=0, k2=0, k3=0, p1=0, p2=0,
                               projection='rectilinear', distortion_type=''):
        """
        Create a camera from manually entered parameters without calibration.

        Args:
            image_size: Tuple of (width_px, height_px)
            location: astropy.coordinates.EarthLocation object for ECEF camera generation
            fx, fy: Focal lengths in pixels
            cx, cy: Camera center in pixels
            azimuth: Camera pointing azimuth in degrees (0=North)
            elevation: Camera elevation angle in degrees (0=horizon, 90=zenith)
            rotation: Camera roll rotation in degrees
            k1, k2, k3: Distortion coefficients
            projection: Projection type ('rectilinear', 'equirectangular', 'stereographic')

        Returns:
            Camera instance with manually set parameters
        """
        # Create new camera instance
        camera = cls(image_size)
        
        # Get projection class
        projection_class = cameraprojections.projection_by_name(projection)
        
        # Estimate sensor size and focal length in mm for projection setup
        # Use standard 35mm equivalent assumptions
        sensor_size = (36, 24)  # Standard 35mm sensor size
        focal_length_mm = fx * sensor_size[0] / image_size[0]  # Convert px to mm
        
        # Create camera_enu with manual parameters
        distortion_class = distortion_by_name(distortion_type)
        camera.camera_enu = ct.Camera(
            projection_class(focallength_mm=focal_length_mm, sensor=sensor_size, image=image_size),
            ct.SpatialOrientation(elevation_m=0),
            distortion_class()
        )
        
        # Set manual parameters
        camera.camera_enu.focallength_x_px = fx
        camera.camera_enu.focallength_y_px = fy
        camera.camera_enu.center_x_px = cx
        camera.camera_enu.center_y_px = cy
        
        # Set orientation (convert elevation to tilt: tilt = elevation + 90)
        camera.camera_enu.heading_deg = azimuth
        camera.camera_enu.tilt_deg = elevation + 90
        camera.camera_enu.roll_deg = rotation
        
        # Set position
        camera.camera_enu.pos_x_m = 0
        camera.camera_enu.pos_y_m = 0
        camera.camera_enu.elevation_m = 0
        
        # Set distortion parameters
        camera.camera_enu.k1 = k1
        camera.camera_enu.k2 = k2
        camera.camera_enu.k3 = k3
        camera.camera_enu.p1 = p1
        camera.camera_enu.p2 = p2

        # Create ECEF camera from ENU camera using provided location
        camera.camera_ecef = camera.camera_ecef_from_camera_enu(camera.camera_enu, location)

        return camera
        
    def __getstate__(self):
        state = {}
        # Save image_size
        state['image_size'] = self.image_size

        for ckey, cam in [('ENU',self.camera_enu), ('ECEF',self.camera_ecef)]:
            if cam is not None:
                keys = cam.parameters.parameters.keys()
                state[ckey] = {key: getattr(cam, key) for key in keys}
                state[ckey]['projectiontype'] =  cameraprojections.name_by_projection(cam.projection)
                state[ckey]['distortiontype'] =  name_by_distortion(cam.lens)
        return state
        
    def __setstate__(self, state):
        if 'camera_enu' in state:
            # fallback to default pickle saved with previous versions
            self.__dict__ = state
        else:
            # Restore image_size
            self.image_size = state.get('image_size', None)

            self.camera_enu = None
            self.camera_ecef = None
            for ckey in ['ENU','ECEF']:
                if ckey in state:
                    if 'projectiontype' in state[ckey]:
                        projection = cameraprojections.projection_by_name(state[ckey]['projectiontype'])
                    else:
                        projection = ct.RectilinearProjection
                    distortion_class = distortion_by_name(state[ckey].get('distortiontype',None))
                    cam = ct.Camera(projection(),
                          ct.SpatialOrientation(),
                          distortion_class())
                    for key in state[ckey]:
                        if key not in ['projectiontype', 'distortiontype']:
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

    def Fit(self, star_references, location, obs_time, distortion=0, centers=True, separate_x_y=True, 
            projectiontype='rectilinear', focallength_35mm=None):
        """
        Fit camera parameters using star calibration data.

        Args:
            star_references: List of StarReference objects containing pixel and sky coordinates
            location: astropy.coordinates.EarthLocation object for observation location
            obs_time: astropy.time.Time object for observation time
            distortion: Order of lens distortion correction (0-3)
            centers: Whether to optimize camera center position
            separate_x_y: Whether to use separate focal lengths for X and Y axes
            projectiontype: Type of camera projection model
            focallength_35mm: Optional focal length in mm (35mm equivalent). If None, defaults to 24mm.
        """
        sensor_size = (36, 24)

        # Use provided focal length or default to 24mm
        focallength = 24 if focallength_35mm is None else focallength_35mm

        # if focal length given specify range f/2 - 2*f, otherwise use wide range
        f_bounds = np.array([5, 600]) if (focallength_35mm is None or focallength_35mm <= 0.0) else np.array([focallength/2.0, focallength*2.0])
        f_bounds = f_bounds/sensor_size[0]*self.image_size[0]

        focallength_px=focallength/sensor_size[0]*self.image_size[0]

        image_size = self.image_size

        cx_bounds = [0, self.image_size[0]]
        cy_bounds = [0, self.image_size[1]]
        projection=cameraprojections.projection_by_name(projectiontype)


        if distortion >= 4:
            from sudrabainiemakoni.cv2_lens_distortion import OpenCVBrownLensDistortion
            distortion_class = OpenCVBrownLensDistortion
        else:
            from sudrabainiemakoni.lensdistortions import BrownLensDistortionLimited
            distortion_class = BrownLensDistortionLimited
        
        print('Projection type:', projection)
        print('Distortion type:', distortion_class)

        self.camera_enu = ct.Camera(projection(focallength_mm=focallength,
                                         sensor=sensor_size,
                                         image=image_size),
                  ct.SpatialOrientation(elevation_m=0),
                  distortion_class())

        # Create AltAz frame for coordinate transformation
        import astropy.coordinates
        from sudrabainiemakoni.starreference import get_stars_enu_unit_coords
        altaz_frame = astropy.coordinates.AltAz(location=location, obstime=obs_time)

        # Extract ENU coordinates and pixel coordinates from star references
        enu_unit_coords = get_stars_enu_unit_coords(star_references, altaz_frame)
        pxls = np.array([r.pixelcoords for r in star_references])
        self.camera_enu.addLandmarkInformation(pxls, enu_unit_coords, [0.01, 0.01, 0.01])
        self.camera_enu.pos_x_m=0
        self.camera_enu.pos_y_m=0
        self.camera_enu.elevation_m=0
        if len(pxls)<6:
            self.camera_enu = optimize_camera.OptimizeCamera(self.camera_enu, enu_unit_coords, pxls, 
                                                            distortion=distortion, centers=centers,separate_x_y=separate_x_y,
                                                            f_bounds=f_bounds, cx_bounds=cx_bounds, cy_bounds=cy_bounds
                                                            )
        else:
            from optimize_camera_cv2 import optimize_camera_cv2
            self.camera_enu = optimize_camera_cv2(self.camera_enu, enu_unit_coords, pxls, 
                                                            distortion=distortion, centers=centers,separate_x_y=separate_x_y,                                                         
                                                            )

        calculated_star_px_coords = self.camera_enu.imageFromSpace(enu_unit_coords)
        residuals = calculated_star_px_coords - pxls
        print('ENU camera res:',np.sqrt(np.mean(residuals**2)))
        for sr, delta in zip(star_references, residuals):
            print(sr, np.abs(delta))

        self.camera_ecef =  self.camera_ecef_from_camera_enu(self.camera_enu, location)

        ecef_unit = pymap3d.enu2ecef(enu_unit_coords[:,0], enu_unit_coords[:,1], enu_unit_coords[:,2],
                                      location.lat.value, location.lon.value, location.height.value)
        ecef_unit =np.array(ecef_unit).T

        print('ECEF camera res:',np.sqrt(np.mean((self.camera_ecef.imageFromSpace(ecef_unit)-pxls)**2)))

        return

    def camera_ecef_from_camera_enu(self, camera_enu, location):
        import copy
        # ja lieto astropy objektus, tad pārveidot no radiāniem uz grādiem nevajag
        sinlon, coslon = np.sin(location.lon).value, np.cos(location.lon).value
        sinlat, coslat = np.sin(location.lat).value, np.cos(location.lat).value
        rotMatr_uvw_enu=np.array([
            [-sinlon, coslon, 0],
            [-sinlat*coslon, -sinlat*sinlon, coslat],
            [coslat*coslon, coslat*sinlon, sinlat]
        ])
        angl = optimize_camera.Orientation_fromRotation(optimize_camera.Rotation_fromOrientation(camera_enu)*Rotation.from_matrix(rotMatr_uvw_enu))

        camera_ecef = ct.Camera(copy.deepcopy(camera_enu.projection),
                                ct.SpatialOrientation(**angl),
                                copy.deepcopy(camera_enu.lens))
        camera_ecef.pos_x_m=location.x.value
        camera_ecef.pos_y_m=location.y.value
        camera_ecef.elevation_m=location.z.value
        return camera_ecef

    def calculate_residuals(self, star_references, location, obs_time):
        """
        Calculate calibration residuals from current camera parameters.

        Args:
            star_references: List of StarReference objects
            location: astropy.coordinates.EarthLocation
            obs_time: astropy.time.Time

        Returns:
            dict with keys:
                - 'star_pixel_coords': Actual digitized positions (N, 2)
                - 'model_pixel_coords': Model-predicted positions (N, 2)
                - 'residuals': model - actual (N, 2)
                - 'rms': Root mean square of residuals
        """
        return calculate_residuals(self.camera_enu, star_references, location, obs_time)

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
    def make_cameras(cls, lat_deg, lon_deg, height_m, width_px, height_px,
                     focallength_35mm, azimuth, elevation, rotation,
                     projectiontype='rectilinear',
                     distortion_type=''):
        projection=cameraprojections.projection_by_name(projectiontype)
        distortion_class=distortion_by_name(distortion_type)
        camera_enu = ct.Camera(projection(focallength_mm=focallength_35mm,
                                             sensor=(36,24),
                                             image=(width_px, height_px)),
                               ct.SpatialOrientation(
                                   heading_deg = azimuth,
                                   tilt_deg = 90 + elevation,
                                   roll_deg = rotation,
                                   elevation_m=0.0
                               ),  distortion_class())
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


def calculate_residuals(camera_enu, star_references, location, obs_time):
    """
    Calculate calibration residuals from camera parameters.

    Args:
        camera_enu: cameratransform.Camera object in ENU coordinate system
        star_references: List of StarReference objects
        location: astropy.coordinates.EarthLocation
        obs_time: astropy.time.Time

    Returns:
        dict with keys:
            - 'star_pixel_coords': Actual digitized positions (N, 2)
            - 'model_pixel_coords': Model-predicted positions (N, 2)
            - 'residuals': model - actual (N, 2)
            - 'rms': Root mean square of residuals
    """
    if camera_enu is None:
        print("No camera calibration available")
        return None

    # Create AltAz frame for coordinate transformation
    import astropy.coordinates
    from sudrabainiemakoni.starreference import get_stars_enu_unit_coords
    altaz_frame = astropy.coordinates.AltAz(location=location, obstime=obs_time)

    # Extract ENU coordinates and pixel coordinates from star references
    enu_unit_coords = get_stars_enu_unit_coords(star_references, altaz_frame)
    star_pixel_coords = np.array([r.pixelcoords for r in star_references])

    # Calculate model predictions
    model_pixel_coords = camera_enu.imageFromSpace(enu_unit_coords)

    # Calculate residuals (model - actual)
    residuals = model_pixel_coords - star_pixel_coords
    rms = np.sqrt(np.mean(residuals**2))

    return {
        'star_pixel_coords': star_pixel_coords,
        'model_pixel_coords': model_pixel_coords,
        'residuals': residuals,
        'rms': rms
    }