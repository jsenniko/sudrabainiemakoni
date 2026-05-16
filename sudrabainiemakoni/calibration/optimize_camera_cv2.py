import numpy as np
from scipy.optimize import least_squares
import cameratransform as ct
from scipy.spatial.transform import Rotation


def Orientation_fromRotation(rr: Rotation):
    eul = rr.as_euler('ZXZ', True)
    return {"roll_deg": -eul[0], "tilt_deg": -eul[1], "heading_deg": eul[2]}


def optimize_camera_cv2(camera, enu_unit_coords, pxls, distortion=3, focallength=True, centers=True, separate_x_y=True, optimize_rotation=True):
    import cv2

    # print(f"\n{'='*60}")
    # print(f"DEBUG: optimize_camera_cv2 called")
    # print(f"  Optimization flags: distortion={distortion}, optimize_rotation={optimize_rotation}, focallength={focallength}, centers={centers}, separate_x_y={separate_x_y}")
    # print(f"  Input camera parameters:")
    # print(f"    Rotation: heading={camera.heading_deg:.2f}°, tilt={camera.tilt_deg:.2f}°, roll={camera.roll_deg:.2f}°")
    # print(f"    Intrinsics: fx={camera.focallength_x_px:.2f}, fy={camera.focallength_y_px:.2f}, cx={camera.center_x_px:.2f}, cy={camera.center_y_px:.2f}")
    # print(f"    Distortion: k1={getattr(camera, 'k1', 0.0):.6f}, k2={getattr(camera, 'k2', 0.0):.6f}, k3={getattr(camera, 'k3', 0.0):.6f}, p1={getattr(camera, 'p1', 0.0):.6f}, p2={getattr(camera, 'p2', 0.0):.6f}")
    # print(f"{'='*60}\n")

    def residuals(params, enu_pts, px_obs, camera_ref, distortion_level, opt_focal, opt_centers, opt_sep_xy, opt_rotation):
        """
        Compute reprojection residuals with variable parameter vector.
        Parameter order matches the flags set in the calling function.
        """
        n = 0

        # Extract rotation if optimizing
        if opt_rotation:
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
        if distortion_level is not None:
            k1 = k2 = k3 = p1 = p2 = k4 = k5 = k6 = 0.0
            if distortion_level >= 1 and distortion_level!=6:
                k1 = params[n]
                n += 1
                if distortion_level >= 2 and distortion_level!=6:
                    k2 = params[n]
                    n += 1
                    if distortion_level >= 3 and distortion_level!=6:
                        k3 = params[n]
                        n += 1

            if distortion_level == 4:
                p1 = params[n]
                p2 = params[n+1]
                n += 2

            if distortion_level >= 5:
                k4 = params[n]
                k5 = params[n+1]
                k6 = params[n+2]
                n += 3
        else:
            # Use camera's current distortion coefficients
            k1 = getattr(camera_ref, 'k1', 0.0)
            k2 = getattr(camera_ref, 'k2', 0.0)
            k3 = getattr(camera_ref, 'k3', 0.0)
            p1 = getattr(camera_ref, 'p1', 0.0)
            p2 = getattr(camera_ref, 'p2', 0.0)
            k4 = getattr(camera_ref, 'k4', 0.0)
            k5 = getattr(camera_ref, 'k5', 0.0)
            k6 = getattr(camera_ref, 'k6', 0.0)

        # -fx to maintain compatibility with cameratransform
        K = np.array([[-fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

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

    if optimize_rotation:
        x0.extend(rvec0.ravel())

    if focallength:
        x0.append(fx0)
        if separate_x_y:
            x0.append(fy0)

    if centers:
        x0.extend([cx0, cy0])

    # Radial distortion parameters (k1, k2, k3)
    if distortion is not None and distortion >= 1 and distortion!=6:
        x0.append(0.0)
        if distortion >= 2 and distortion!=6:
            x0.append(0.0)
            if distortion >= 3 and distortion!=6:
                x0.append(0.0)

    # Tangential distortion parameters (p1, p2)
    if distortion is not None and distortion == 4:
        x0.extend([0.0, 0.0])

    # Rational distortion parameters (k4, k5, k6)
    if distortion is not None and distortion >= 5:
        x0.extend([0.0, 0.0, 0.0])

    x0 = np.array(x0)

    result = least_squares(
        residuals,
        x0,
        args=(enu, px, camera, distortion, focallength, centers, separate_x_y, optimize_rotation),
        method='lm',
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000
    )

    # Unpack results based on flags
    n = 0

    if optimize_rotation:
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

    if distortion is not None:
        k1 = k2 = k3 = p1 = p2 = k4 = k5 = k6 = 0.0
        if distortion >= 1 and distortion!=6:
            k1 = result.x[n]
            n += 1
            if distortion >= 2 and distortion!=6:
                k2 = result.x[n]
                n += 1
                if distortion >= 3 and distortion!=6:
                    k3 = result.x[n]
                    n += 1

        if distortion == 4:
            p1 = result.x[n]
            p2 = result.x[n+1]
            n += 2

        if distortion >= 5:
            k4 = result.x[n]
            k5 = result.x[n+1]
            k6 = result.x[n+2]
            n += 3
    else:
        # Keep existing distortion coefficients
        k1 = getattr(camera, 'k1', 0.0)
        k2 = getattr(camera, 'k2', 0.0)
        k3 = getattr(camera, 'k3', 0.0)
        p1 = getattr(camera, 'p1', 0.0)
        p2 = getattr(camera, 'p2', 0.0)
        k4 = getattr(camera, 'k4', 0.0)
        k5 = getattr(camera, 'k5', 0.0)
        k6 = getattr(camera, 'k6', 0.0)

    #print(f"\nDEBUG: Unpacked optimization results:")
    #print(f"  Rotation optimized: {optimize_rotation}")
    rr_result = Rotation.from_rotvec(rvec_opt)
    angles_result = Orientation_fromRotation(rr_result)
    print(f"    heading={angles_result['heading_deg']:.2f}°, tilt={angles_result['tilt_deg']:.2f}°, roll={angles_result['roll_deg']:.2f}°")
    print(f"  Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Distortion: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")

    # Calculate RMS error
    res = result.fun.reshape(-1, 2)
    rms = np.sqrt(np.mean(res**2))

    print(f"Converged: {result.success}  |  message: {result.message}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    if distortion is not None and distortion >= 1 and distortion!=6:
        print(f"k1={k1:.6f}", end="")
        if distortion >= 2 and distortion!=6:
            print(f"  k2={k2:.6f}", end="")
            if distortion >= 3 and distortion!=6:
                print(f"  k3={k3:.6f}", end="")
        if distortion == 4:
            print(f"  p1={p1:.6f}  p2={p2:.6f}", end="")
        if distortion >= 5:
            print(f"  k4={k4:.6f}  k5={k5:.6f}  k6={k6:.6f}", end="")
        print()
    print(f"rvec = {rvec_opt}")

    # Determine distortion class
    if distortion is None:
        # Keep existing distortion class
        distortion_class = type(camera.lens)
    elif distortion == 4:
        try:
            from sudrabainiemakoni.calibration.lens_distortions.cv2_lens_distortion import OpenCVBrownLensDistortion
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lens_distortions'))
            from cv2_lens_distortion import OpenCVBrownLensDistortion
        distortion_class = OpenCVBrownLensDistortion
    elif distortion in [5,6]:
        try:
            from sudrabainiemakoni.calibration.lens_distortions.lensdistortions import RationalDistortionLimited
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lens_distortions'))
            from lensdistortions import RationalDistortionLimited
        distortion_class = RationalDistortionLimited
    else:
        try:
            from sudrabainiemakoni.calibration.lens_distortions.lensdistortions import BrownLensDistortionLimited
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lens_distortions'))
            from lensdistortions import BrownLensDistortionLimited
        distortion_class = BrownLensDistortionLimited

    # Re-pack to cameratransform
    # print(f"\nDEBUG: Creating new camera with parameters:")
    # print(f"  Rotation to set: heading={angles_result['heading_deg']:.2f}°, tilt={angles_result['tilt_deg']:.2f}°, roll={angles_result['roll_deg']:.2f}°")
    # print(f"  Intrinsics to set: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    # print(f"  Distortion to set: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")

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

    # Always set distortion coefficients (either optimized or preserved from camera)
    if hasattr(camnew, 'k1'):
        camnew.k1 = k1
    if hasattr(camnew, 'k2'):
        camnew.k2 = k2
    if hasattr(camnew, 'k3'):
        camnew.k3 = k3
    if hasattr(camnew, 'p1'):
        camnew.p1 = p1
    if hasattr(camnew, 'p2'):
        camnew.p2 = p2
    if hasattr(camnew, 'k4'):
        camnew.k4 = k4
    if hasattr(camnew, 'k5'):
        camnew.k5 = k5
    if hasattr(camnew, 'k6'):
        camnew.k6 = k6

    # print(f"\nDEBUG: Final camera parameters after assignment:")
    # print(f"  Rotation: heading={camnew.heading_deg:.2f}°, tilt={camnew.tilt_deg:.2f}°, roll={camnew.roll_deg:.2f}°")
    # print(f"  Intrinsics: fx={camnew.focallength_x_px:.2f}, fy={camnew.focallength_y_px:.2f}, cx={camnew.center_x_px:.2f}, cy={camnew.center_y_px:.2f}")
    # print(f"  Distortion: k1={getattr(camnew, 'k1', 0.0):.6f}, k2={getattr(camnew, 'k2', 0.0):.6f}, k3={getattr(camnew, 'k3', 0.0):.6f}, p1={getattr(camnew, 'p1', 0.0):.6f}, p2={getattr(camnew, 'p2', 0.0):.6f}")
    # print(f"{'='*60}\n")
    return camnew
