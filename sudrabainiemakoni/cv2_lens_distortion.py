import numpy as np
import cv2
import cameratransform
from cameratransform.parameter_set import ParameterSet, Parameter, TYPE_DISTORTION


class OpenCVBrownLensDistortion(cameratransform.lens_distortion.LensDistortion):
    """
    Brown lens distortion using OpenCV (cv2) for distortion calculations.

    This class uses both radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.
    It uses cv2's implementation for both forward and inverse distortion.

    Compatible with cameratransform for camera projection.
    """

    def __init__(self, k1=None, k2=None, k3=None, p1=None, p2=None, projection=None):
        """
        Initialize Brown lens distortion model with radial and tangential coefficients.

        Parameters:
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
            projection: Camera projection object (set later via setProjection)
        """
        self.parameters = ParameterSet(
            k1=Parameter(k1, default=0, range=(None, None), type=TYPE_DISTORTION),
            k2=Parameter(k2, default=0, range=(None, None), type=TYPE_DISTORTION),
            k3=Parameter(k3, default=0, range=(None, None), type=TYPE_DISTORTION),
            p1=Parameter(p1, default=0, range=(None, None), type=TYPE_DISTORTION),
            p2=Parameter(p2, default=0, range=(None, None), type=TYPE_DISTORTION),
        )
        # Set callbacks to update matrices when parameters change
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._update_matrices
        self.projection = projection
        self._update_matrices()

    def setProjection(self, projection):
        """Set the camera projection and update internal matrices."""
        self.projection = projection
        # Merge distortion parameters with projection parameters
        self.parameters = ParameterSet(
            k1=self.parameters.parameters["k1"],
            k2=self.parameters.parameters["k2"],
            k3=self.parameters.parameters["k3"],
            p1=self.parameters.parameters["p1"],
            p2=self.parameters.parameters["p2"],
            image_width_px=self.projection.parameters.parameters["image_width_px"],
            image_height_px=self.projection.parameters.parameters["image_height_px"],
            focallength_x_px=self.projection.parameters.parameters["focallength_x_px"],
            focallength_y_px=self.projection.parameters.parameters["focallength_y_px"],
            center_x_px=self.projection.parameters.parameters["center_x_px"],
            center_y_px=self.projection.parameters.parameters["center_y_px"],
        )
        # Set callbacks to update matrices when any parameter changes
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._update_matrices
        self._update_matrices()

    def _update_matrices(self):
        """Update camera matrix and distortion coefficients for cv2."""
        if self.projection is not None:
            # Build OpenCV camera matrix
            self.camera_matrix = np.array([
                [self.projection.focallength_x_px, 0, self.projection.center_x_px],
                [0, self.projection.focallength_y_px, self.projection.center_y_px],
                [0, 0, 1]
            ], dtype=np.float64)

            # Update scale and offset for compatibility
            self.scale = np.array([self.projection.focallength_x_px,
                                   self.projection.focallength_y_px])
            self.offset = np.array([self.projection.center_x_px,
                                    self.projection.center_y_px])
        else:
            # Default identity camera matrix
            self.camera_matrix = np.eye(3, dtype=np.float64)
            self.scale = np.array([1.0, 1.0])
            self.offset = np.array([0.0, 0.0])

        # Build distortion coefficient vector [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([
            self.parameters.k1,
            self.parameters.k2,
            self.parameters.p1,
            self.parameters.p2,
            self.parameters.k3
        ], dtype=np.float64)

    def imageFromDistorted(self, points):
        """
        Convert distorted image points to undistorted image points.

        This is the inverse distortion operation (undistortion).
        Uses cv2.undistortPoints internally.

        Parameters:
            points: Array of distorted points, shape (..., 2)

        Returns:
            Array of undistorted points, same shape as input
        """
        points = np.array(points, dtype=np.float64)
        original_shape = points.shape

        # Reshape to (N, 1, 2) as required by cv2.undistortPoints
        points_reshaped = points.reshape(-1, 1, 2)

        # Undistort using cv2
        # P=camera_matrix ensures output is in pixel coordinates
        undistorted = cv2.undistortPoints(
            points_reshaped,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix
        )

        # Reshape back to original shape
        return undistorted.reshape(original_shape)

    def distortedFromImage(self, points):
        """
        Convert undistorted image points to distorted image points.

        This is the forward distortion operation.
        Uses cv2.projectPoints internally.

        Parameters:
            points: Array of undistorted points, shape (..., 2)

        Returns:
            Array of distorted points, same shape as input
        """
        points = np.array(points, dtype=np.float64)
        original_shape = points.shape

        # Convert image points to normalized coordinates
        points_normalized = (points.reshape(-1, 2) - self.offset) / self.scale

        # Create 3D points at unit depth (Z=1) for projectPoints
        points_3d = np.column_stack([
            points_normalized[:, 0],
            points_normalized[:, 1],
            np.ones(len(points_normalized))
        ]).astype(np.float64)

        # Project with distortion using identity rotation and translation
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)

        distorted, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        # Reshape back to original shape
        return distorted.reshape(original_shape)
