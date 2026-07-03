import numpy as np
from scipy.optimize import minimize
import cameratransform as ct
from scipy.spatial.transform import Rotation


def Orientation_fromRotation(rr: Rotation):
    eul = rr.as_euler('ZXZ', True)
    return {"roll_deg": -eul[0], "tilt_deg": -eul[1], "heading_deg": eul[2]}


def optimize_camera_minimize(camera, enu_unit_coords, pxls, distortion=3, focallength=True, centers=True, separate_x_y=True, optimize_rotation=True,
                             monotonic_distortion=True):
    import cv2

    def residuals_vec(params, enu_pts, px_obs, camera_ref, distortion_level, opt_focal, opt_centers, opt_sep_xy, opt_rotation):
        n = 0

        if opt_rotation:
            rvec = params[n:n+3].reshape(3, 1)
            n += 3
        else:
            rr = Rotation.from_euler('ZXZ', [-camera_ref.roll_deg, -camera_ref.tilt_deg, camera_ref.heading_deg], degrees=True)
            rvec = rr.as_rotvec().reshape(3, 1)

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

        if opt_centers:
            cx = params[n]
            cy = params[n+1]
            n += 2
        else:
            cx = camera_ref.center_x_px
            cy = camera_ref.center_y_px

        if distortion_level is not None:
            k1 = k2 = k3 = p1 = p2 = k4 = k5 = k6 = 0.0
            if distortion_level >= 1:
                k1 = params[n]; n += 1
                if distortion_level >= 2:
                    k2 = params[n]; n += 1
                    if distortion_level >= 3:
                        k3 = params[n]; n += 1
            if distortion_level >= 4:
                p1 = params[n]; p2 = params[n+1]; n += 2
            if distortion_level >= 5:
                k4 = params[n]; k5 = params[n+1]; k6 = params[n+2]; n += 3
        else:
            k1 = getattr(camera_ref, 'k1', 0.0)
            k2 = getattr(camera_ref, 'k2', 0.0)
            k3 = getattr(camera_ref, 'k3', 0.0)
            p1 = getattr(camera_ref, 'p1', 0.0)
            p2 = getattr(camera_ref, 'p2', 0.0)
            k4 = getattr(camera_ref, 'k4', 0.0)
            k5 = getattr(camera_ref, 'k5', 0.0)
            k6 = getattr(camera_ref, 'k6', 0.0)

        K = np.array([[-fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        proj, _ = cv2.projectPoints(
            enu_pts.reshape(-1, 1, 3),
            rvec, tvec, K, dist
        )
        return (proj.reshape(-1, 2) - px_obs).ravel()

    def objective(params, *args):
        r = residuals_vec(params, *args)
        val = np.dot(r, r)
        return val / obj_scale[0]

    # ── Input data ───────────────────────────────────────────────────────────────
    w, h = camera.image_width_px, camera.image_height_px
    fx0, fy0 = camera.focallength_x_px, camera.focallength_y_px
    cx0, cy0 = camera.center_x_px, camera.center_y_px

    enu = enu_unit_coords.astype(np.float64)
    px = pxls.astype(np.float64)

    # Initial rotation estimate via solvePnP
    K0 = np.array([[-fx0, 0, cx0],
                   [0, fy0, cy0],
                   [0, 0, 1]], dtype=np.float64)
    d0 = np.zeros((4, 1), dtype=np.float64)
    ok, rvec0, _ = cv2.solvePnP(enu, px, K0, d0, flags=cv2.SOLVEPNP_ITERATIVE)

    # Build initial parameter vector
    x0 = []
    if optimize_rotation:
        x0.extend(rvec0.ravel())
    if focallength:
        x0.append(fx0)
        if separate_x_y:
            x0.append(fy0)
    if centers:
        x0.extend([cx0, cy0])
    if distortion is not None and distortion >= 1:
        x0.append(0.0)
        if distortion >= 2:
            x0.append(0.0)
            if distortion >= 3:
                x0.append(0.0)
    if distortion is not None and distortion >= 4:
        x0.extend([0.0, 0.0])
    if distortion is not None and distortion >= 5:
        x0.extend([0.0, 0.0, 0.0])
    x0 = np.array(x0)

    args = (enu, px, camera, distortion, focallength, centers, separate_x_y, optimize_rotation)

    # ── Monotonicity constraints ──────────────────────────────────────────────────
    # OpenCV radial distortion: r_d = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
    # Derivative w.r.t. r:  f'(r) = 1 + 3*k1*r^2 + 5*k2*r^4 + 7*k3*r^6
    # Monotonicity: all f'(r) >= 0  OR  all f'(r) <= 0 over [0, r_max].
    # f'(0) = 1 > 0 always, so for a non-inverted physical lens: all >= 0.
    # r is in normalised coords (pixels / focal_length).
    # r_max = farthest corner from principal point, checked across all four corners.

    constraints = []
    idx_k1 = idx_k2 = idx_k3 = None
    if monotonic_distortion and distortion is not None and distortion >= 1:
        # Locate parameter indices in x0
        n_tmp = 0
        if optimize_rotation:
            n_tmp += 3
        if focallength:
            n_tmp += 2 if separate_x_y else 1
        if centers:
            n_tmp += 2
        idx_k1 = n_tmp if distortion >= 1 else None
        idx_k2 = (n_tmp + 1) if distortion >= 2 else None
        idx_k3 = (n_tmp + 2) if distortion >= 3 else None

        # Focal/center indices for computing r_max from current params
        n_rot = 3 if optimize_rotation else 0
        idx_fx = n_rot if focallength else None
        idx_fy = (n_rot + 1) if (focallength and separate_x_y) else None
        idx_cx = (n_rot + (2 if separate_x_y else 1)) if (focallength and centers) else (n_rot if centers else None)
        idx_cy = (idx_cx + 1) if (centers and idx_cx is not None) else None

        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)

        def current_r_max(params):
            fx_ = params[idx_fx] if idx_fx is not None else fx0
            fy_ = params[idx_fy] if idx_fy is not None else fy0
            cx_ = params[idx_cx] if idx_cx is not None else cx0
            cy_ = params[idx_cy] if idx_cy is not None else cy0
            dx = (corners[:, 0] - cx_) / fx_
            dy = (corners[:, 1] - cy_) / fy_
            return np.sqrt(dx**2 + dy**2).max()

        def count_positive_real_roots(params):
            """
            Count positive real roots of r*(1 + k1*r^2 + k2*r^4 + k3*r^6) = r_d_max.
            Polynomial (degree 7): k3*r^7 + k2*r^5 + k1*r^3 + r - r_d_max = 0
            Coefficients for numpy.roots (highest degree first):
              [k3, 0, k2, 0, k1, 0, 1, -r_d_max]
            A monotone mapping has exactly one positive real root.
            A non-monotone mapping has two (or more).
            """
            r_d_max = current_r_max(params)
            k1_ = params[idx_k1] if idx_k1 is not None else 0.0
            k2_ = params[idx_k2] if idx_k2 is not None else 0.0
            k3_ = params[idx_k3] if idx_k3 is not None else 0.0
            coeffs = [k3_, 0.0, k2_, 0.0, k1_, 0.0, 1.0, -r_d_max]
            roots = np.roots(coeffs)
            positive_real = roots[np.abs(roots.imag) < 1e-8 * np.abs(roots.real + 1e-30)].real
            return int(np.sum(positive_real > 1e-10))

        con_call_count = [0]

        def monotonic_con(params):
            n_roots = count_positive_real_roots(params)
            val = 0.5 if n_roots == 1 else -0.5  # >= 0 iff exactly 1 positive real root
            k1_ = params[idx_k1] if idx_k1 is not None else 0.0
            k2_ = params[idx_k2] if idx_k2 is not None else 0.0
            k3_ = params[idx_k3] if idx_k3 is not None else 0.0
            r_dm = current_r_max(params)
            con_call_count[0] += 1
            print(f"  [con #{con_call_count[0]:4d}] k1={k1_:.6f} k2={k2_:.6f} k3={k3_:.6f}  r_max={r_dm:.4f}  n_roots={n_roots}  con={val:.1f}")
            return val

        constraints = [{'type': 'ineq', 'fun': monotonic_con}]

        print(f"Monotonicity constraint: root-count based, r_max(init)={current_r_max(x0):.4f}")
        print(f"  Initial root count: {count_positive_real_roots(x0)}")

    # Scale objective to ~1.0 at x0 to improve SLSQP QP subproblem conditioning
    _obj0 = np.dot(residuals_vec(x0, *args), residuals_vec(x0, *args))
    obj_scale = [max(_obj0, 1.0)]

    iter_count = [0]
    def callback(xk):
        iter_count[0] += 1
        obj_val = objective(xk, *args)
        if monotonic_distortion and constraints:
            k1_ = xk[idx_k1] if idx_k1 is not None else 0.0
            k2_ = xk[idx_k2] if idx_k2 is not None else 0.0
            k3_ = xk[idx_k3] if idx_k3 is not None else 0.0
            n_roots = count_positive_real_roots(xk)
            print(f"[iter {iter_count[0]:3d}] obj={obj_val:.6f}  k1={k1_:.6f} k2={k2_:.6f} k3={k3_:.6f}  n_roots={n_roots}")
        else:
            print(f"[iter {iter_count[0]:3d}] obj={obj_val:.6f}")

    result = minimize(
        objective,
        x0,
        args=args,
        method='SLSQP',
        constraints=tuple(constraints),
        options={'ftol': 1e-12, 'maxiter': 1000},
        callback=callback,
    )

    if monotonic_distortion and constraints:
        r_max_sol = current_r_max(result.x)
        n_roots_sol = count_positive_real_roots(result.x)
        print(f"  Post-solve: r_max={r_max_sol:.4f}  root_count={n_roots_sol}  ({'monotone' if n_roots_sol <= 1 else 'NON-MONOTONE'})")
        print(f"  Total constraint calls: {con_call_count[0]}")

    # Unpack results
    n = 0
    if optimize_rotation:
        rvec_opt = result.x[n:n+3]
        n += 3
    else:
        rr = Rotation.from_euler('ZXZ', [-camera.roll_deg, -camera.tilt_deg, camera.heading_deg], degrees=True)
        rvec_opt = rr.as_rotvec()

    if focallength:
        if separate_x_y:
            fx = result.x[n]; fy = result.x[n+1]; n += 2
        else:
            fx = fy = result.x[n]; n += 1
    else:
        fx = camera.focallength_x_px
        fy = camera.focallength_y_px

    if centers:
        cx = result.x[n]; cy = result.x[n+1]; n += 2
    else:
        cx = camera.center_x_px
        cy = camera.center_y_px

    if distortion is not None:
        k1 = k2 = k3 = p1 = p2 = k4 = k5 = k6 = 0.0
        if distortion >= 1:
            k1 = result.x[n]; n += 1
            if distortion >= 2:
                k2 = result.x[n]; n += 1
                if distortion >= 3:
                    k3 = result.x[n]; n += 1
        if distortion >= 4:
            p1 = result.x[n]; p2 = result.x[n+1]; n += 2
        if distortion >= 5:
            k4 = result.x[n]; k5 = result.x[n+1]; k6 = result.x[n+2]; n += 3
    else:
        k1 = getattr(camera, 'k1', 0.0)
        k2 = getattr(camera, 'k2', 0.0)
        k3 = getattr(camera, 'k3', 0.0)
        p1 = getattr(camera, 'p1', 0.0)
        p2 = getattr(camera, 'p2', 0.0)
        k4 = getattr(camera, 'k4', 0.0)
        k5 = getattr(camera, 'k5', 0.0)
        k6 = getattr(camera, 'k6', 0.0)

    rr_result = Rotation.from_rotvec(rvec_opt)
    angles_result = Orientation_fromRotation(rr_result)
    print(f"    heading={angles_result['heading_deg']:.2f}, tilt={angles_result['tilt_deg']:.2f}, roll={angles_result['roll_deg']:.2f}")
    print(f"  Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Distortion: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")

    res_vec = residuals_vec(result.x, *args)
    res2 = res_vec.reshape(-1, 2)
    rms = np.sqrt(np.mean(res2**2))
    print(f"Converged: {result.success}  |  message: {result.message}")
    print(f"RMS reprojection error: {rms:.4f} px")

    # Determine distortion class
    if distortion is None:
        distortion_class = type(camera.lens)
    elif distortion >= 4:
        try:
            from sudrabainiemakoni.calibration.lens_distortions.cv2_lens_distortion import OpenCVBrownLensDistortion
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lens_distortions'))
            from cv2_lens_distortion import OpenCVBrownLensDistortion
        distortion_class = OpenCVBrownLensDistortion
    else:
        try:
            from sudrabainiemakoni.calibration.lens_distortions.lensdistortions import BrownLensDistortionLimited
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lens_distortions'))
            from lensdistortions import BrownLensDistortionLimited
        distortion_class = BrownLensDistortionLimited

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

    if hasattr(camnew, 'k1'): camnew.k1 = k1
    if hasattr(camnew, 'k2'): camnew.k2 = k2
    if hasattr(camnew, 'k3'): camnew.k3 = k3
    if hasattr(camnew, 'p1'): camnew.p1 = p1
    if hasattr(camnew, 'p2'): camnew.p2 = p2
    if hasattr(camnew, 'k4'): camnew.k4 = k4
    if hasattr(camnew, 'k5'): camnew.k5 = k5
    if hasattr(camnew, 'k6'): camnew.k6 = k6

    return camnew
