import numpy as np
from scipy.optimize import least_squares
import cameratransform as ct
from scipy.spatial.transform import Rotation
from sudrabainiemakoni.optimize_camera import Orientation_fromRotation

#def OptimizeCamera(camera, enu_unit_coords, pxls, distortion=3, focallength = True, centers=True,  separate_x_y=True, fixed_rotation=False,
#                   f_bounds=[500,10000], cx_bounds=[0, 6000], cy_bounds=[0,4000]):
def optimize_camera_cv2(camera, enu_unit_coords, pxls, distortion=3, focallength=True, centers=True, separate_x_y=True, fixed_rotation=False):
    import cv2

    def residuals(params, enu_pts, px_obs, camera_ref, distortion_level, opt_focal, opt_centers, opt_sep_xy, opt_fixed_rot):
        """
        Compute reprojection residuals with variable parameter vector.
        Parameter order matches the flags set in the calling function.
        """
        n = 0

        # Extract rotation if not fixed
        if not opt_fixed_rot:
            rvec = params[n:n+3].reshape(3, 1)
            n += 3
        else:
            # Use camera's current rotation
            rr = Rotation.from_euler('ZXZ', [-camera_ref.roll_deg, -camera_ref.tilt_deg, camera_ref.heading_deg], degrees=True)
            rvec = rr.as_rotvec().reshape(3, 1)

        # Extract focal lengths
        if opt_focal:
            if opt_sep_xy:
                fx = params[n]
                fy = params[n+1]
                n += 2
            else:
                fx = fy = params[n]
                n += 1
        else:
            fx = camera_ref.focallength_x_px
            fy = camera_ref.focallength_y_px

        # Extract center coordinates
        if opt_centers:
            cx = params[n]
            cy = params[n+1]
            n += 2
        else:
            cx = camera_ref.center_x_px
            cy = camera_ref.center_y_px

        # Extract distortion parameters
        k1 = k2 = k3 = p1 = p2 = 0.0
        if distortion_level >= 1:
            k1 = params[n]
            n += 1
            if distortion_level >= 2:
                k2 = params[n]
                n += 1
                if distortion_level >= 3:
                    k3 = params[n]
                    n += 1

        if distortion_level >= 4:
            p1 = params[n]
            p2 = params[n+1]
            n += 2

        # -fx to maintain compatibility with cameratransform
        K = np.array([[-fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        tvec = np.zeros((3, 1), dtype=np.float64)

        proj, _ = cv2.projectPoints(
            enu_pts.reshape(-1, 1, 3),
            rvec, tvec, K, dist
        )
        proj = proj.reshape(-1, 2)

        return (proj - px_obs).ravel()
    # ── Input data ───────────────────────────────────────────────────────────────
    w, h = camera.image_width_px, camera.image_height_px
    fx0, fy0 = camera.focallength_x_px, camera.focallength_y_px
    cx0, cy0 = camera.center_x_px, camera.center_y_px

    enu = enu_unit_coords.astype(np.float64)
    px = pxls.astype(np.float64)

    # Initial K for solvePnP
    K0 = np.array([[-fx0, 0, cx0],
                   [0, fy0, cy0],
                   [0, 0, 1]], dtype=np.float64)
    d0 = np.zeros((4, 1), dtype=np.float64)

    # Step 1 - Initial rotation estimate
    ok, rvec0, _ = cv2.solvePnP(
        enu, px, K0, d0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Build initial parameter vector based on flags
    x0 = []

    if not fixed_rotation:
        x0.extend(rvec0.ravel())

    if focallength:
        x0.append(fx0)
        if separate_x_y:
            x0.append(fy0)

    if centers:
        x0.extend([cx0, cy0])

    # Radial distortion parameters (k1, k2, k3)
    if distortion >= 1:
        x0.append(0.0)
        if distortion >= 2:
            x0.append(0.0)
            if distortion >= 3:
                x0.append(0.0)

    # Tangential distortion parameters (p1, p2)
    if distortion >= 4:
        x0.extend([0.0, 0.0])

    x0 = np.array(x0)

    result = least_squares(
        residuals,
        x0,
        args=(enu, px, camera, distortion, focallength, centers, separate_x_y, fixed_rotation),
        method='lm',
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000
    )

    # Unpack results based on flags
    n = 0

    if not fixed_rotation:
        rvec_opt = result.x[n:n+3]
        n += 3
    else:
        rr = Rotation.from_euler('ZXZ', [-camera.roll_deg, -camera.tilt_deg, camera.heading_deg], degrees=True)
        rvec_opt = rr.as_rotvec()

    if focallength:
        if separate_x_y:
            fx = result.x[n]
            fy = result.x[n+1]
            n += 2
        else:
            fx = fy = result.x[n]
            n += 1
    else:
        fx = camera.focallength_x_px
        fy = camera.focallength_y_px

    if centers:
        cx = result.x[n]
        cy = result.x[n+1]
        n += 2
    else:
        cx = camera.center_x_px
        cy = camera.center_y_px

    k1 = k2 = k3 = p1 = p2 = 0.0
    if distortion >= 1:
        k1 = result.x[n]
        n += 1
        if distortion >= 2:
            k2 = result.x[n]
            n += 1
            if distortion >= 3:
                k3 = result.x[n]
                n += 1

    if distortion >= 4:
        p1 = result.x[n]
        p2 = result.x[n+1]
        n += 2


    # Calculate RMS error
    res = result.fun.reshape(-1, 2)
    rms = np.sqrt(np.mean(res**2))

    print(f"Converged: {result.success}  |  message: {result.message}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    if distortion >= 1:
        print(f"k1={k1:.6f}", end="")
        if distortion >= 2:
            print(f"  k2={k2:.6f}", end="")
            if distortion >= 3:
                print(f"  k3={k3:.6f}", end="")
        if distortion >= 4:
            print(f"  p1={p1:.6f}  p2={p2:.6f}", end="")
        print()
    print(f"rvec = {rvec_opt}")

    # Determine distortion class
    if distortion >= 4:
        from sudrabainiemakoni.cv2_lens_distortion import OpenCVBrownLensDistortion
        distortion_class = OpenCVBrownLensDistortion
    else:
        from sudrabainiemakoni.lensdistortions import BrownLensDistortionLimited
        distortion_class = BrownLensDistortionLimited

    # Re-pack to cameratransform
    camnew = ct.Camera(ct.RectilinearProjection(), ct.SpatialOrientation(elevation_m=0.0), distortion_class())
    rr = Rotation.from_rotvec(rvec_opt)
    cv2_cam = Orientation_fromRotation(rr)
    camnew.image_width_px = w
    camnew.image_height_px = h
    camnew.focallength_x_px = fx
    camnew.focallength_y_px = fy
    camnew.center_x_px = cx
    camnew.center_y_px = cy
    camnew.roll_deg = cv2_cam['roll_deg']
    camnew.tilt_deg = cv2_cam['tilt_deg']
    camnew.heading_deg = cv2_cam['heading_deg']

    if distortion >= 1:
        camnew.k1 = k1
        if distortion >= 2:
            camnew.k2 = k2
            if distortion >= 3:
                camnew.k3 = k3

    if distortion >= 4:
        camnew.p1 = p1
        camnew.p2 = p2

    return camnew