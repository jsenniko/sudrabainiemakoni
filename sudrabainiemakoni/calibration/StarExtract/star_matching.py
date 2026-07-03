import copy
import numpy as np
import pymap3d
from scipy.spatial import KDTree

try:
    from sudrabainiemakoni.calibration.catalog_star_projection import project_catalog_stars
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from catalog_star_projection import project_catalog_stars


class StarMatchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angular_distances(vecs):
    """NxN matrix of pairwise angular distances in degrees from unit vectors."""
    dots = np.clip(vecs @ vecs.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _catalog_enu(catalog_df):
    """Convert catalog altitude/azimuth columns to ENU unit vectors."""
    cat_e, cat_n, cat_u = pymap3d.aer2enu(
        catalog_df['azimuth'].values,
        catalog_df['altitude'].values,
        1.0,
    )
    vecs = np.column_stack([cat_e, cat_n, cat_u])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _run_optimizer(camera_enu, enu_coords, pxls,
                   focallength=False, centers=False, separate_x_y=True):
    """
    Run optimize_camera_cv2 with rotation always on, distortion kept as-is.
    Intrinsics (focallength, centers) are optional.
    Returns a new camera_enu preserving projection type and lens from input camera.
    """
    try:
        from sudrabainiemakoni.calibration.optimize_camera_cv2 import optimize_camera_cv2
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from optimize_camera_cv2 import optimize_camera_cv2

    cam_opt = optimize_camera_cv2(
        camera_enu,
        enu_coords,
        pxls,
        distortion=None,
        optimize_focallength=focallength,
        centers=centers,
        separate_x_y=separate_x_y,
        optimize_rotation=True,
    )

    # optimize_camera_cv2 always creates RectilinearProjection; restore original
    # projection type and lens from the input camera, copy only the fitted values
    cam_restored = copy.deepcopy(camera_enu)
    cam_restored.heading_deg = cam_opt.heading_deg
    cam_restored.tilt_deg    = cam_opt.tilt_deg
    cam_restored.roll_deg    = cam_opt.roll_deg
    if focallength:
        cam_restored.focallength_x_px = cam_opt.focallength_x_px
        cam_restored.focallength_y_px = cam_opt.focallength_y_px
    if centers:
        cam_restored.center_x_px = cam_opt.center_x_px
        cam_restored.center_y_px = cam_opt.center_y_px
    return cam_restored


def _compute_residuals(camera_enu, enu_coords, pxls):
    """
    Reproject enu_coords through camera and return per-star pixel residuals.
    Returns residuals array (N, 2) and RMS over inliers (stars below outlier threshold).
    """
    projected = camera_enu.imageFromSpace(enu_coords)
    residuals = projected - pxls          # (N, 2)
    per_star  = np.linalg.norm(residuals, axis=1)   # (N,)
    # outlier threshold: 3x median
    median_r  = np.median(per_star)
    inlier    = per_star < max(3.0 * median_r, 3.0)
    rms       = float(np.sqrt(np.mean(per_star[inlier] ** 2)))
    return residuals, per_star, inlier, rms


# ---------------------------------------------------------------------------
# Step 1 – triangle-based angular matching (ratio-based, focal-length invariant)
# EXPERIMENTAL: not working reliably — wrong matches accumulate comparable vote
# counts to correct ones when the catalog contains ~50+ stars. Needs quads or
# a better discrimination method before production use.
# ---------------------------------------------------------------------------

def match_stars_angular_ratio_based(
    detected_xy,
    camera_enu,
    location,
    date,
    max_magnitude=4.0,
    ratio_tol=0.01,
    min_votes=3,
    n_search=50,
):
    """
    Match detected image stars to catalog stars using triangle shape ratios.

    For each triangle of stars, the two independent shape parameters
    (b/a, c/a) where a >= b >= c are the sorted pairwise distances are
    invariant to rotation AND to uniform scale. Because a uniform focal
    length error scales all angular distances by the same factor, these
    ratios cancel it out exactly (for equidistant/fisheye lenses where
    r = f*theta, the scale is perfectly uniform). Only centroiding noise
    and distortion residuals affect the ratios, so the tolerance can be
    kept tight (~0.01-0.05) regardless of focal length accuracy.

    Both sides use angular distances:
    - Detected: back-projected via camera.getRay() (affected by fl error
      in absolute value, but ratios within a triangle are fl-invariant)
    - Catalog: computed from alt/az ENU vectors (exact sky geometry)

    Parameters
    ----------
    detected_xy : array (N, 2)
        Detected star pixel coordinates, sorted brightest-first.
    camera_enu : cameratransform Camera
    location : astropy EarthLocation
    date : astropy Time
    max_magnitude : float
    ratio_tol : float
        Tolerance on triangle side ratios b/a and c/a (dimensionless).
        0.02 corresponds to ~2% shape difference. Driven by centroiding
        noise and lens non-linearity, not focal length error.
    min_votes : int
        Minimum triangle votes to accept a match.
    n_search : int
        Number of brightest detected stars used in triangle search.

    Returns
    -------
    matches : list of (detected_idx, catalog_idx, votes)
        Sorted by votes descending, above automatic 10%-of-max threshold.
    detected_enu : ndarray (N, 3)
    catalog_df : DataFrame
    catalog_enu : ndarray (Nc, 3)
    """
    _, detected_enu = camera_enu.getRay(detected_xy, normed=True)

    catalog_df = project_catalog_stars(camera_enu, location, date, max_magnitude=max_magnitude)
    cat_enu    = _catalog_enu(catalog_df)

    N  = min(len(detected_enu), n_search)
    Nc = len(cat_enu)

    det_ang = _angular_distances(detected_enu[:N])
    cat_ang = _angular_distances(cat_enu)

    # Pre-compute shape ratios for all catalog triplets: (r1=b/a, r2=c/a)
    # and store the vertex indices in sorted-side order.
    # For triplet (p,q,r), sides are cat_ang[p,q], cat_ang[p,r], cat_ang[q,r].
    # Sorting sides descending gives the longest (a) and we label which
    # vertex is opposite which side.
    #
    # For a triangle with vertices (p, q, r):
    #   side opposite p  = cat_ang[q, r]
    #   side opposite q  = cat_ang[p, r]
    #   side opposite r  = cat_ang[p, q]
    # The longest side a is opposite the vertex we call v0; b opposite v1; c opposite v2.
    # When we match detected (i,j,k) to catalog (p,q,r) with the same side ordering,
    # the vertex opposite the longest side in detected maps to the vertex opposite
    # the longest side in catalog — giving us the star-to-star assignment.

    # Pre-compute catalog triplet shape ratios and vertex assignments once.
    # For triplet (p,q,r): side opposite p = cat_ang[q,r], etc.
    # Sort sides descending; store r1=b/a, r2=c/a and the three opposite-vertices.
    n_triplets = Nc * (Nc - 1) * (Nc - 2) // 6
    cat_r1 = np.empty(n_triplets)
    cat_r2 = np.empty(n_triplets)
    cat_va = np.empty(n_triplets, dtype=np.int32)
    cat_vb = np.empty(n_triplets, dtype=np.int32)
    cat_vc = np.empty(n_triplets, dtype=np.int32)
    t = 0
    for p in range(Nc):
        for q in range(p + 1, Nc):
            for r in range(q + 1, Nc):
                s0 = cat_ang[q, r]  # opposite p
                s1 = cat_ang[p, r]  # opposite q
                s2 = cat_ang[p, q]  # opposite r
                if s0 >= s1 and s0 >= s2:
                    a, va = s0, p
                    if s1 >= s2: b, vb, c, vc = s1, q, s2, r
                    else:        b, vb, c, vc = s2, r, s1, q
                elif s1 >= s0 and s1 >= s2:
                    a, va = s1, q
                    if s0 >= s2: b, vb, c, vc = s0, p, s2, r
                    else:        b, vb, c, vc = s2, r, s0, p
                else:
                    a, va = s2, r
                    if s0 >= s1: b, vb, c, vc = s0, p, s1, q
                    else:        b, vb, c, vc = s1, q, s0, p
                cat_r1[t] = b / a if a > 1e-6 else -1.0
                cat_r2[t] = c / a if a > 1e-6 else -1.0
                cat_va[t] = va; cat_vb[t] = vb; cat_vc[t] = vc
                t += 1
    valid_cat = cat_r1 >= 0

    vote = np.zeros((N, Nc), dtype=np.int32)

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                s0 = det_ang[j, k]  # opposite i
                s1 = det_ang[i, k]  # opposite j
                s2 = det_ang[i, j]  # opposite k
                if s0 >= s1 and s0 >= s2:
                    da, dva = s0, i
                    if s1 >= s2: db, dvb, dc, dvc = s1, j, s2, k
                    else:        db, dvb, dc, dvc = s2, k, s1, j
                elif s1 >= s0 and s1 >= s2:
                    da, dva = s1, j
                    if s0 >= s2: db, dvb, dc, dvc = s0, i, s2, k
                    else:        db, dvb, dc, dvc = s2, k, s0, i
                else:
                    da, dva = s2, k
                    if s0 >= s1: db, dvb, dc, dvc = s0, i, s1, j
                    else:        db, dvb, dc, dvc = s1, j, s0, i
                if da < 1e-6:
                    continue
                d_r1 = db / da
                d_r2 = dc / da

                mask = (valid_cat &
                        (np.abs(cat_r1 - d_r1) < ratio_tol) &
                        (np.abs(cat_r2 - d_r2) < ratio_tol))
                for h in np.where(mask)[0]:
                    vote[dva, cat_va[h]] += 1
                    vote[dvb, cat_vb[h]] += 1
                    vote[dvc, cat_vc[h]] += 1

    raw_matches  = []
    assigned_cat = set()
    for di in range(N):
        best_ci = int(np.argmax(vote[di]))
        best_v  = int(vote[di, best_ci])
        if best_v >= min_votes and best_ci not in assigned_cat:
            raw_matches.append((di, best_ci, best_v))
            assigned_cat.add(best_ci)

    raw_matches.sort(key=lambda x: -x[2])

    if raw_matches:
        top_votes = raw_matches[0][2]
        threshold = max(min_votes, top_votes * 0.10)
        matches   = [(di, ci, v) for di, ci, v in raw_matches if v >= threshold]
    else:
        matches = []

    return matches, detected_enu, catalog_df, cat_enu


# ---------------------------------------------------------------------------
# Step 1 – triangle-based angular matching (absolute angular distances)
# Requires the initial focal length to be sufficiently accurate (~0.3% or better).
# A 1% focal length error causes ~0.3° distance error on 30° pairs, which is
# the same order as the matching tolerance and leads to wrong matches.
# ---------------------------------------------------------------------------

def match_stars_angular(
    detected_xy,
    camera_enu,
    location,
    date,
    max_magnitude=4.0,
    angle_tol_deg=0.1,
    focal_length_uncertainty=0.003,
    min_votes=3,
    n_search=50,
):
    """
    Match detected image stars to catalog stars using pairwise angular distances.

    Angular distances between stars are invariant to camera rotation, so this
    works even when the camera pose is significantly wrong.

    Parameters
    ----------
    detected_xy : array (N, 2)
        Detected star pixel coordinates, sorted brightest-first.
    camera_enu : cameratransform Camera
    location : astropy EarthLocation
    date : astropy Time
    max_magnitude : float
    angle_tol_deg : float
        Base tolerance for matching angular distance triangle sides (degrees).
    focal_length_uncertainty : float
        Fractional focal length uncertainty (e.g. 0.01 = 1%). Widens the
        per-pair tolerance proportionally to each pair's angular separation,
        absorbing back-projection errors from imprecise focal length.
        Pairs longer than angle_tol_deg / focal_length_uncertainty are skipped
        to keep false-match rate low.
    min_votes : int
        Minimum triangle votes to accept a match.
    n_search : int
        Number of brightest detected stars used in triangle search.

    Returns
    -------
    matches : list of (detected_idx, catalog_idx, votes)
        Sorted by votes descending, above automatic 10%-of-max threshold.
    detected_enu : ndarray (N, 3)
    catalog_df : DataFrame  (name, mag, altitude, azimuth, pixel_x, pixel_y, ...)
    catalog_enu : ndarray (Nc, 3)
    """
    _, detected_enu = camera_enu.getRay(detected_xy, normed=True)

    catalog_df  = project_catalog_stars(camera_enu, location, date, max_magnitude=max_magnitude)
    cat_enu     = _catalog_enu(catalog_df)

    N  = min(len(detected_enu), n_search)
    Nc = len(cat_enu)

    det_ang = _angular_distances(detected_enu[:N])
    cat_ang = _angular_distances(cat_enu)

    # Precompute flat catalog pair arrays (a < b) sorted by distance for fast lookup.
    # Shape: (P,) where P = Nc*(Nc-1)/2
    pa, pb = np.triu_indices(Nc, k=1)
    pair_dist = cat_ang[pa, pb]           # distance for each catalog pair
    sort_order = np.argsort(pair_dist)
    pair_dist_s = pair_dist[sort_order]   # sorted distances
    pa_s = pa[sort_order]                 # catalog index a (sorted)
    pb_s = pb[sort_order]                 # catalog index b (sorted)

    vote = np.zeros((N, Nc), dtype=np.int32)

    for i in range(N):
        for j in range(i + 1, N):
            dij = det_ang[i, j]
            tol_ij = angle_tol_deg + dij * focal_length_uncertainty

            # Binary search: catalog pairs whose distance is within tol of dij
            lo = np.searchsorted(pair_dist_s, dij - tol_ij)
            hi = np.searchsorted(pair_dist_s, dij + tol_ij)
            if lo >= hi:
                continue
            ca_ij = pa_s[lo:hi]   # shape (M,)
            cb_ij = pb_s[lo:hi]   # shape (M,)

            for k in range(j + 1, N):
                dik = det_ang[i, k]
                djk = det_ang[j, k]
                tol_ik = angle_tol_deg + dik * focal_length_uncertainty
                tol_jk = angle_tol_deg + djk * focal_length_uncertainty

                # For all candidate pairs (a,b) simultaneously, check both orderings.
                # Ordering 1: i->a, j->b — need third star c s.t. dist(a,c)~dik, dist(b,c)~djk
                # Ordering 2: i->b, j->a — need third star c s.t. dist(b,c)~dik, dist(a,c)~djk
                # cat_ang[ca_ij, :] has shape (M, Nc); broadcast over all c candidates at once.

                d_ac = cat_ang[ca_ij, :]   # (M, Nc)
                d_bc = cat_ang[cb_ij, :]   # (M, Nc)

                # Ordering 1: i->a, j->b
                valid1 = (np.abs(d_ac - dik) < tol_ik) & (np.abs(d_bc - djk) < tol_jk)
                # Exclude c==a or c==b
                c_idx = np.arange(Nc)
                valid1 &= (c_idx[None, :] != ca_ij[:, None]) & (c_idx[None, :] != cb_ij[:, None])
                m_pairs, m_c = np.where(valid1)
                if len(m_pairs):
                    np.add.at(vote, (i, ca_ij[m_pairs]), 1)
                    np.add.at(vote, (j, cb_ij[m_pairs]), 1)
                    np.add.at(vote, (k, m_c), 1)

                # Ordering 2: i->b, j->a
                valid2 = (np.abs(d_bc - dik) < tol_ik) & (np.abs(d_ac - djk) < tol_jk)
                valid2 &= (c_idx[None, :] != ca_ij[:, None]) & (c_idx[None, :] != cb_ij[:, None])
                m_pairs2, m_c2 = np.where(valid2)
                if len(m_pairs2):
                    np.add.at(vote, (i, cb_ij[m_pairs2]), 1)
                    np.add.at(vote, (j, ca_ij[m_pairs2]), 1)
                    np.add.at(vote, (k, m_c2), 1)

    raw_matches  = []
    assigned_cat = set()
    for di in range(N):
        best_ci = int(np.argmax(vote[di]))
        best_v  = int(vote[di, best_ci])
        if best_v >= min_votes and best_ci not in assigned_cat:
            raw_matches.append((di, best_ci, best_v))
            assigned_cat.add(best_ci)

    raw_matches.sort(key=lambda x: -x[2])

    if raw_matches:
        top_votes = raw_matches[0][2]
        threshold = max(min_votes, top_votes * 0.10)
        matches   = [(di, ci, v) for di, ci, v in raw_matches if v >= threshold]
    else:
        matches = []

    return matches, detected_enu, catalog_df, cat_enu


# ---------------------------------------------------------------------------
# Step 3 – nearest-neighbour matching after pose correction
# ---------------------------------------------------------------------------

def match_stars_nn(
    detected_xy,
    camera_enu,
    location,
    date,
    max_magnitude=5.0,
    max_dist_px=15.0,
    debug_star=None,
):
    """
    Match detected stars to catalog stars by nearest neighbour in pixel space.
    Only valid after camera pose has been approximately corrected.

    Parameters
    ----------
    debug_star : str or None
        If set, print diagnostic info for this catalog star name at every stage.

    Returns
    -------
    matched_detected_xy : ndarray (M, 2)
    matched_catalog_enu : ndarray (M, 3)
    matched_catalog_rows : DataFrame (M rows, same columns as catalog_df)
    catalog_df : DataFrame  (all in-view catalog stars)
    residuals_px : ndarray (M,)  per-star pixel distance
    """
    catalog_df = project_catalog_stars(camera_enu, location, date, max_magnitude=max_magnitude)
    cat_xy     = catalog_df[['pixel_x', 'pixel_y']].values
    cat_enu    = _catalog_enu(catalog_df)

    if debug_star is not None:
        try:
            from sudrabainiemakoni.calibration.catalog_star_projection import project_catalog_stars_debug
        except ImportError:
            from catalog_star_projection import project_catalog_stars_debug
        project_catalog_stars_debug(
            camera_enu, location, date,
            debug_names=debug_star,
            max_magnitude=max_magnitude,
            skip_pixel_filter=True,
        )
        rows = catalog_df[catalog_df['name'] == debug_star]
        if not rows.empty:
            r = rows.iloc[0]
            proj_xy = np.array([r['pixel_x'], r['pixel_y']])
            det_dists = np.linalg.norm(detected_xy - proj_xy, axis=1)
            nearest_idx = int(np.argmin(det_dists))
            nearest_dist = det_dists[nearest_idx]
            print(f"  [debug {debug_star}] nearest detected star: idx={nearest_idx}, "
                  f"pos=({detected_xy[nearest_idx,0]:.1f}, {detected_xy[nearest_idx,1]:.1f}), "
                  f"dist={nearest_dist:.1f} px  (max_dist_px={max_dist_px})")
            if nearest_dist >= max_dist_px:
                print(f"  [debug {debug_star}] -> MISSED: nearest detected star is {nearest_dist:.1f} px away, exceeds max_dist_px={max_dist_px}")
            else:
                print(f"  [debug {debug_star}] -> will be matched (dist within threshold)")

    tree = KDTree(cat_xy)
    dists, idxs = tree.query(detected_xy, distance_upper_bound=max_dist_px)

    valid = dists < max_dist_px

    # Deduplicate: if multiple detected stars matched the same catalog star,
    # keep only the one with the smallest distance.
    valid_det   = np.where(valid)[0]
    valid_cat   = idxs[valid]
    valid_dists = dists[valid]

    best_for_cat = {}  # cat_idx -> (det_idx, dist)
    for pos, (det_i, cat_i, d) in enumerate(zip(valid_det, valid_cat, valid_dists)):
        if cat_i not in best_for_cat or d < best_for_cat[cat_i][1]:
            best_for_cat[cat_i] = (det_i, d)

    kept_det   = np.array([v[0] for v in best_for_cat.values()], dtype=int)
    kept_cat   = np.array(list(best_for_cat.keys()), dtype=int)
    kept_dists = np.array([v[1] for v in best_for_cat.values()])

    n_dupes = valid.sum() - len(kept_det)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate detection(s) (multiple detections -> same catalog star)")

    matched_detected_xy  = detected_xy[kept_det]
    matched_catalog_enu  = cat_enu[kept_cat]
    matched_catalog_rows = catalog_df.iloc[kept_cat].reset_index(drop=True)
    residuals_px         = kept_dists

    return matched_detected_xy, matched_catalog_enu, matched_catalog_rows, catalog_df, residuals_px


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def calibrate_camera_pose(
    detected_xy,
    camera_enu,
    location,
    date,
    # angular matching parameters
    max_magnitude_coarse=4.0,
    angle_tol_deg=0.1,
    focal_length_uncertainty=0.003,
    min_votes=3,
    n_search=50,
    # nn matching parameters
    max_magnitude_fine=5.0,
    nn_max_dist_px=5.0,
    # optimizer flags (rotation always on; distortion always kept as-is)
    optimize_intrinsics=True,
    # residual thresholds
    max_rms_coarse_px=5.0,
    max_rms_fine_px=3.0,
    # debug
    debug_star=None,
):
    """
    Full star-matching and pose refinement pipeline.

    Steps
    -----
    1. Angular triangle matching to get initial correspondences.
    1a. Optimize rotation only; check coarse residual.
    2. Re-project catalog with corrected camera.
    3. Nearest-neighbour matching to collect more stars.
    3a. Check fine residual.
    4. Re-optimize rotation only on larger match set.
    5. Return refined camera and final matched star positions.

    Parameters
    ----------
    detected_xy : ndarray (N, 2)
        Detected star pixel coordinates, sorted brightest-first.
    camera_enu : cameratransform Camera
        Initial (possibly inaccurate) camera.
    location : astropy EarthLocation
    date : astropy Time
    max_magnitude_coarse : float
        Catalog magnitude limit for angular matching.
    angle_tol_deg : float
        Angular distance tolerance for triangle matching.
    min_votes : int
        Minimum triangle votes to accept a coarse match.
    n_search : int
        Detected stars used in triangle search.
    max_magnitude_fine : float
        Catalog magnitude limit for NN matching (can be fainter).
    nn_max_dist_px : float
        Maximum pixel distance for NN matching.
    max_rms_coarse_px : float
        Maximum allowed RMS after coarse optimization; raises StarMatchError if exceeded.
    max_rms_fine_px : float
        Maximum allowed RMS after fine optimization; raises StarMatchError if exceeded.

    Returns
    -------
    refined_camera : cameratransform Camera
        Camera with corrected pose (distortion/intrinsics unchanged).
    final_matches : dict with keys:
        'detected_xy'  : ndarray (M, 2)
        'catalog_enu'  : ndarray (M, 3)
        'catalog_df'   : DataFrame of all catalog stars in view
        'residuals_px' : ndarray (M,)  per-star pixel residual
        'rms_px'       : float
    """
    MIN_STARS = 6

    # ------------------------------------------------------------------
    # Step 1 – coarse angular matching
    # ------------------------------------------------------------------
    print("Step 1: Angular triangle matching...")
    matches, detected_enu, catalog_df_coarse, catalog_enu_coarse = match_stars_angular(
        detected_xy, camera_enu, location, date,
        max_magnitude=max_magnitude_coarse,
        angle_tol_deg=angle_tol_deg,
        focal_length_uncertainty=focal_length_uncertainty,
        min_votes=min_votes,
        n_search=n_search,
    )
    print(f"  Coarse matches: {len(matches)}")
    if len(matches) < MIN_STARS:
        raise StarMatchError(
            f"Too few coarse matches: {len(matches)} (need >= {MIN_STARS}). "
            "Try increasing angle_tol_deg or max_magnitude_coarse."
        )

    det_idxs = np.array([m[0] for m in matches])
    cat_idxs = np.array([m[1] for m in matches])
    vote_counts = np.array([m[2] for m in matches])
    enu_coords_coarse = catalog_enu_coarse[cat_idxs]
    pxls_coarse       = detected_xy[det_idxs]

    # ------------------------------------------------------------------
    # Step 1a – coarse optimization using only top-voted matches as seed.
    # Low-vote matches are more likely to be false positives; seeding the
    # optimizer with only the highest-confidence subset avoids divergence.
    # ------------------------------------------------------------------
    print("Step 1a: Coarse rotation optimization...")
    vote_thresh_seed = max(min_votes, int(np.median(vote_counts)))
    seed_mask = vote_counts >= vote_thresh_seed
    if seed_mask.sum() < MIN_STARS:
        seed_mask = np.ones(len(matches), dtype=bool)
    print(f"  Using {seed_mask.sum()}/{len(matches)} top-voted matches as seed (votes >= {vote_thresh_seed})")
    cam_coarse = _run_optimizer(camera_enu,
                                enu_coords_coarse[seed_mask], pxls_coarse[seed_mask],
                                focallength=optimize_intrinsics, centers=False)

    _, per_star_coarse, inliers_coarse, rms_coarse = _compute_residuals(
        cam_coarse, enu_coords_coarse, pxls_coarse
    )
    print(f"  Coarse RMS (inliers): {rms_coarse:.2f} px  "
          f"({inliers_coarse.sum()}/{len(inliers_coarse)} inliers)")
    for i, (di, ci, v) in enumerate(matches):
        flag = "" if inliers_coarse[i] else "  [outlier]"
        #print(f"    {catalog_df_coarse.iloc[ci]['name']:15s}  "
        #      f"res={per_star_coarse[i]:.1f} px{flag}")

    if rms_coarse > max_rms_coarse_px:
        raise StarMatchError(
            f"Coarse RMS too large: {rms_coarse:.2f} px > {max_rms_coarse_px} px"
        )

    # ------------------------------------------------------------------
    # Step 2 – re-project catalog with corrected camera
    # Step 3 – nearest-neighbour matching
    # ------------------------------------------------------------------
    print("Step 3: Nearest-neighbour matching with refined camera...")
    matched_det_xy, matched_cat_enu, matched_cat_rows, catalog_df_fine, residuals_nn = match_stars_nn(
        detected_xy, cam_coarse, location, date,
        max_magnitude=max_magnitude_fine,
        max_dist_px=nn_max_dist_px,
        debug_star=debug_star,
    )
    print(f"  NN matches: {len(matched_det_xy)}")

    if len(matched_det_xy) < MIN_STARS:
        raise StarMatchError(
            f"Too few NN matches: {len(matched_det_xy)} (need >= {MIN_STARS}). "
            "Try increasing nn_max_dist_px or max_magnitude_fine."
        )

    _, per_star_nn, inliers_nn, rms_nn = _compute_residuals(
        cam_coarse, matched_cat_enu, matched_det_xy
    )
    # NN acceptance already capped at nn_max_dist_px — don't let the 3×median
    # outlier threshold silently reject stars that passed the NN distance check.
    inliers_nn = inliers_nn | (per_star_nn < nn_max_dist_px)
    print(f"  NN RMS before re-optimization (inliers): {rms_nn:.2f} px")


    # ------------------------------------------------------------------
    # Step 3a – filter NN outliers before final optimization
    # ------------------------------------------------------------------
    if debug_star is not None and debug_star in matched_cat_rows['name'].values:
        idx = matched_cat_rows.index[matched_cat_rows['name'] == debug_star][0]
        res = per_star_nn[idx]
        kept = bool(inliers_nn[idx])
        print(f"  [debug {debug_star}] after NN match: res={res:.1f} px, "
              f"inlier={'YES' if kept else 'NO (FILTERED by step 3a outlier removal)'}")

    matched_det_xy_in   = matched_det_xy[inliers_nn]
    matched_cat_enu_in  = matched_cat_enu[inliers_nn]
    matched_cat_rows_in = matched_cat_rows[inliers_nn].reset_index(drop=True)
    residuals_nn_in     = residuals_nn[inliers_nn]


    if len(matched_det_xy_in) < MIN_STARS:
        raise StarMatchError(
            f"Too few NN inliers after outlier removal: {len(matched_det_xy_in)}"
        )

    # ------------------------------------------------------------------
    # Step 4 – fine optimization (rotation only, larger star set)
    # ------------------------------------------------------------------
    print("Step 4: Fine optimization (rotation + intrinsics)...")
    refined_camera = _run_optimizer(cam_coarse, matched_cat_enu_in, matched_det_xy_in,
                                    focallength=optimize_intrinsics,
                                    centers=optimize_intrinsics)

    _, per_star_fine, inliers_fine, rms_fine = _compute_residuals(
        refined_camera, matched_cat_enu_in, matched_det_xy_in
    )
    # Keep any star whose NN match was within nn_max_dist_px — the 3×median
    # outlier threshold must not override an explicit acceptance criterion.
    inliers_fine = inliers_fine | (residuals_nn_in < nn_max_dist_px)

    # TODO actualize catalog_df_fine according to efined_camera, e.g. pixel_x, pixel_y
    print(f"  Fine RMS (inliers): {rms_fine:.2f} px  "
          f"({inliers_fine.sum()}/{len(inliers_fine)} inliers)")

    if debug_star is not None and debug_star in matched_cat_rows_in['name'].values:
        idx = matched_cat_rows_in.index[matched_cat_rows_in['name'] == debug_star][0]
        res = per_star_fine[idx]
        kept = bool(inliers_fine[idx])
        print(f"  [debug {debug_star}] after fine optimization: res={res:.1f} px, "
              f"inlier={'YES -> in final result' if kept else 'NO (FILTERED by step 5 inlier filter)'}")

    if rms_fine > max_rms_fine_px:
        raise StarMatchError(
            f"Fine RMS too large: {rms_fine:.2f} px > {max_rms_fine_px} px"
        )

    # ------------------------------------------------------------------
    # Step 5 – package results
    # ------------------------------------------------------------------


    final_matches = {
        'detected_xy'  : matched_det_xy_in[inliers_fine],
        'catalog_enu'  : matched_cat_enu_in[inliers_fine],
        'catalog_rows' : matched_cat_rows_in[inliers_fine].reset_index(drop=True),
        'catalog_df'   : catalog_df_fine,
        'residuals_px' : per_star_fine[inliers_fine],
        'rms_px'       : rms_fine,
    }

    return refined_camera, final_matches


# ---------------------------------------------------------------------------
# Blind pose estimation (focal length and distortion taken from camera)
# ---------------------------------------------------------------------------

def estimate_pose_blind(
    detected_xy,
    camera_enu,
    location,
    date,
    max_magnitude_coarse=3.0,
    angle_tol_deg=0.1,
    focal_length_uncertainty=0.003,
    min_votes=4,
    n_search=50,
    min_altitude_deg=-10.0,
    max_rms_px=10.0,
):
    """
    Estimate camera pose without any prior pose knowledge.

    Focal length and distortion coefficients are taken from camera_enu as-is.
    Focal length is optimized together with rotation. Distortion is never modified.

    The all-sky catalog (filtered only by altitude) is used so that no prior
    pointing direction is needed.

    Parameters
    ----------
    detected_xy : ndarray (N, 2)
        Detected star pixel coordinates, sorted brightest-first.
    camera_enu : cameratransform Camera
        Provides image size, focal length estimate, and distortion coefficients.
        Pose (heading/tilt/roll) is ignored.
    location : astropy EarthLocation
    date : astropy Time
    max_magnitude_coarse : float
        Use only bright stars for blind matching (default 3.0).
    angle_tol_deg : float
        Base angular distance tolerance for triangle matching (degrees).
        Automatically widened by focal_length_uncertainty.
    focal_length_uncertainty : float
        Fractional focal length uncertainty (e.g. 0.003 = 0.3%).
        Widens angle_tol_deg to absorb back-projection errors caused by
        imprecise focal length: for a star pair separated by θ degrees,
        the angular distance error is ~θ × focal_length_uncertainty.
        The effective tolerance becomes:
            tol = angle_tol_deg + pair_angle × focal_length_uncertainty
        To keep the false-match rate acceptable, pairs longer than
            angle_tol_deg / focal_length_uncertainty
        are excluded from triangle matching.
        Practical limit: values above ~0.005 (0.5%) will likely fail
        with a sparse bright-star catalog (mag < 3). Use fainter stars
        (max_magnitude_coarse=4.0) if focal length is less well known.
    min_votes : int
        Minimum triangle votes to accept a match.
    n_search : int
        Number of brightest detected stars used in triangle search.
    min_altitude_deg : float
        Minimum star altitude to include in the all-sky catalog.
    max_rms_px : float
        Maximum allowed RMS after blind optimization; raises StarMatchError.

    Returns
    -------
    camera_with_pose : cameratransform Camera
        Copy of input camera with estimated pose and focal length.
        Distortion coefficients are unchanged.
    matches : list of (detected_idx, catalog_idx, votes)
    catalog_df : DataFrame  all-sky catalog used for matching
    catalog_enu : ndarray (Nc, 3)
    """
    MIN_STARS = 6

    _, detected_enu = camera_enu.getRay(detected_xy, normed=True)

    catalog_df = project_catalog_stars(
        camera_enu, location, date,
        max_magnitude=max_magnitude_coarse,
        min_altitude=min_altitude_deg,
        skip_pixel_filter=True,
    )
    print(f"  All-sky catalog stars (mag<{max_magnitude_coarse}, alt>{min_altitude_deg}°): {len(catalog_df)}")

    cat_enu = _catalog_enu(catalog_df)

    N  = min(len(detected_enu), n_search)
    Nc = len(cat_enu)

    det_ang = _angular_distances(detected_enu[:N])
    cat_ang = _angular_distances(cat_enu)

    # With uncertain focal length, back-projected angular distances have error
    # proportional to the pair separation: err ≈ θ × focal_length_uncertainty.
    # To keep tolerance tight (and false-match rate low), restrict triangles to
    # pairs with angular separation below max_pair_angle_deg, where the absolute
    # error stays manageable.
    max_pair_angle_deg = angle_tol_deg / max(focal_length_uncertainty, 1e-6)
    # Cap at a reasonable field-of-view fraction to avoid degenerate triangles
    max_pair_angle_deg = min(max_pair_angle_deg, 30.0)
    effective_tol = angle_tol_deg + max_pair_angle_deg * focal_length_uncertainty
    print(f"  Focal length uncertainty: {focal_length_uncertainty*100:.1f}%  "
          f"-> max pair angle: {max_pair_angle_deg:.1f}°  "
          f"effective tolerance: {effective_tol:.3f}°")

    vote          = np.zeros((N, Nc), dtype=np.int32)
    cat_idx_range = np.arange(Nc)

    for i in range(N):
        for j in range(i + 1, N):
            dij = det_ang[i, j]
            if dij > max_pair_angle_deg:
                continue
            tol_ij  = angle_tol_deg + dij * focal_length_uncertainty
            mask_ij = np.abs(cat_ang - dij) < tol_ij
            ca_ij, cb_ij = np.where(
                mask_ij & (cat_idx_range[:, None] < cat_idx_range[None, :])
            )
            if len(ca_ij) == 0:
                continue

            for k in range(j + 1, N):
                dik = det_ang[i, k]
                djk = det_ang[j, k]
                if dik > max_pair_angle_deg or djk > max_pair_angle_deg:
                    continue
                tol_ik = angle_tol_deg + dik * focal_length_uncertainty
                tol_jk = angle_tol_deg + djk * focal_length_uncertainty

                for idx in range(len(ca_ij)):
                    a, b = ca_ij[idx], cb_ij[idx]

                    ck_ik = np.abs(cat_ang[a, :] - dik) < tol_ik
                    ck_jk = np.abs(cat_ang[b, :] - djk) < tol_jk
                    for c in np.where(ck_ik & ck_jk)[0]:
                        if c != a and c != b:
                            vote[i, a] += 1
                            vote[j, b] += 1
                            vote[k, c] += 1

                    ck_ik2 = np.abs(cat_ang[b, :] - dik) < tol_ik
                    ck_jk2 = np.abs(cat_ang[a, :] - djk) < tol_jk
                    for c in np.where(ck_ik2 & ck_jk2)[0]:
                        if c != a and c != b:
                            vote[i, b] += 1
                            vote[j, a] += 1
                            vote[k, c] += 1

    raw_matches  = []
    assigned_cat = set()
    for di in range(N):
        best_ci = int(np.argmax(vote[di]))
        best_v  = int(vote[di, best_ci])
        if best_v >= min_votes and best_ci not in assigned_cat:
            raw_matches.append((di, best_ci, best_v))
            assigned_cat.add(best_ci)

    raw_matches.sort(key=lambda x: -x[2])

    # In blind mode votes are low (sparse bright catalog vs. full sky).
    # Use absolute min_votes threshold only — no relative 10% cutoff.
    matches = [(di, ci, v) for di, ci, v in raw_matches if v >= min_votes]

    print(f"  Blind matches: {len(matches)} (vote threshold >= {min_votes})")
    if len(matches) < MIN_STARS:
        raise StarMatchError(
            f"Too few blind matches: {len(matches)} (need >= {MIN_STARS}). "
            "Try increasing angle_tol_deg, focal_length_uncertainty, or max_magnitude_coarse."
        )

    det_idxs   = np.array([m[0] for m in matches])
    cat_idxs   = np.array([m[1] for m in matches])
    enu_coords = cat_enu[cat_idxs]
    pxls       = detected_xy[det_idxs]
    vote_counts = np.array([m[2] for m in matches])

    fx_orig = camera_enu.focallength_x_px

    # Iteratively tighten: start with top-voted matches, expand on inliers
    # Start with matches above median vote count to reduce false-match contamination
    vote_thresh = max(min_votes, int(np.median(vote_counts)))
    seed_mask   = vote_counts >= vote_thresh
    if seed_mask.sum() < MIN_STARS:
        seed_mask = np.ones(len(matches), dtype=bool)  # fallback: use all

    camera_with_pose = _run_optimizer(camera_enu,
                                      enu_coords[seed_mask], pxls[seed_mask],
                                      focallength=True, centers=False)

    # Validate: focal length must stay positive and plausible
    fx_new = camera_with_pose.focallength_x_px
    if fx_new <= 0 or fx_new > 3 * fx_orig or fx_new < fx_orig / 3:
        raise StarMatchError(
            f"Optimizer produced implausible focal length: {fx_new:.1f} px "
            f"(input was {fx_orig:.1f} px). Matches are likely wrong. "
            "Try increasing min_votes or reducing focal_length_uncertainty."
        )

    _, per_star, inliers, rms = _compute_residuals(camera_with_pose, enu_coords, pxls)

    # Second pass on inliers across all matches to recover more stars and refine
    if inliers.sum() >= MIN_STARS:
        camera_with_pose = _run_optimizer(camera_with_pose,
                                          enu_coords[inliers], pxls[inliers],
                                          focallength=True, centers=False)
        _, per_star, inliers, rms = _compute_residuals(
            camera_with_pose, enu_coords, pxls)

    print(f"  Blind RMS (inliers): {rms:.2f} px  ({inliers.sum()}/{len(inliers)} inliers)")
    for i, (di, ci, v) in enumerate(matches):
        flag = "" if inliers[i] else "  [outlier]"
        print(f"    {catalog_df.iloc[ci]['name']:15s}  votes={v:4d}  res={per_star[i]:.1f} px{flag}")

    if rms > max_rms_px:
        raise StarMatchError(
            f"Blind pose RMS too large: {rms:.2f} px > {max_rms_px} px. "
            "Focal length estimate may be too far off."
        )

    return camera_with_pose, matches, catalog_df, cat_enu
