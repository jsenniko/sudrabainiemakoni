import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats


def _centroid_and_metrics(gray_image, cx, cy, window, ih, iw):
    half = window // 2
    x1 = max(0, int(round(cx)) - half)
    x2 = min(iw, int(round(cx)) + half + 1)
    y1 = max(0, int(round(cy)) - half)
    y2 = min(ih, int(round(cy)) + half + 1)

    patch = gray_image[y1:y2, x1:x2].copy()
    if patch.size < 4:
        return cx, cy, np.nan, np.nan, np.nan, np.nan

    # Local background: median of border pixels
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    bg = np.median(border)
    data = patch - bg
    data[data < 0] = 0

    total = data.sum()
    if total <= 0:
        return cx, cy, np.nan, np.nan, np.nan, np.nan

    ys, xs = np.mgrid[y1:y2, x1:x2].astype(float)
    refined_x = (xs * data).sum() / total
    refined_y = (ys * data).sum() / total

    peak = data.max()
    flux = total

    # Roundness from second moments
    dx = xs - refined_x
    dy = ys - refined_y
    mxx = (dx * dx * data).sum() / total
    myy = (dy * dy * data).sum() / total
    mxy = (dx * dy * data).sum() / total
    denom = mxx + myy
    roundness = 0.0 if denom <= 0 else (mxx - myy) / denom

    # Sharpness: ratio of peak pixel to average above background
    above = data[data > 0]
    sharpness = float(peak / above.mean()) if len(above) > 0 else np.nan

    return refined_x, refined_y, sharpness, roundness, flux, peak + bg

def detect_stars_grid(image, grid_size=50, upscale_factor=10, min_contrast=10,
                      threshold_factor=0.8, refine_window=5, min_star_area=4,
                      max_star_area=400, grid_overlap=25,
                      use_starfinder=True, starfinder_fwhm=5.0,
                      starfinder_threshold_sigma=3.0, starfinder_sharplo=0.2,
                      starfinder_sharphi=1.0, starfinder_roundlo=-1.0,
                      starfinder_roundhi=1.0,
                      starfinder_local_window=None):

    ih, iw = image.shape[:2]
    candidates = []

    step_size = grid_size - grid_overlap
    if step_size <= 0:
        step_size = grid_size

    grid_starts_x = list(range(0, iw - grid_size + 1, step_size))
    if grid_starts_x[-1] + grid_size < iw:
        grid_starts_x.append(iw - grid_size)

    grid_starts_y = list(range(0, ih - grid_size + 1, step_size))
    if grid_starts_y[-1] + grid_size < ih:
        grid_starts_y.append(ih - grid_size)

    for x1 in grid_starts_x:
        for y1 in grid_starts_y:
            x2 = x1 + grid_size
            y2 = y1 + grid_size
            if x2 > iw:
                x2 = iw
            if y2 > ih:
                y2 = ih

            tile = image[y1:y2, x1:x2]

            if tile.size == 0:
                continue
            upscaled_size_x, upscaled_size_y = (x2-x1) * upscale_factor, (y2-y1) * upscale_factor
            upscaled_tile = cv2.resize(tile, (upscaled_size_x, upscaled_size_y))

            if len(upscaled_tile.shape) == 3:
                upscaled_tile = cv2.cvtColor(upscaled_tile, cv2.COLOR_BGR2GRAY)

            avg_px = np.mean(upscaled_tile)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(upscaled_tile)
            px_diff = max_val - avg_px

            if px_diff > min_contrast:
                thresh_val = max_val * threshold_factor
                _, thresh_img = cv2.threshold(upscaled_tile, thresh_val, 255, cv2.THRESH_BINARY)

                cnt_res = cv2.findContours(thresh_img.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                contours = cnt_res[0] if len(cnt_res) == 2 else cnt_res[1]

                if len(contours) == 1:
                    cnt = contours[0]
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx_local = x + w / 2
                    cy_local = y + h / 2

                    cx = x1 + (cx_local / upscale_factor)
                    cy = y1 + (cy_local / upscale_factor)
                    candidates.append({
                        'x': cx,
                        'y': cy,
                        'intensity': 0,
                        'contrast': px_diff,
                        'area': 0
                    })

    deduplicated = []
    duplicates_removed = []
    for star in candidates:
        is_duplicate = False
        for existing in deduplicated:
            dist = np.sqrt((star['x'] - existing['x'])**2 + (star['y'] - existing['y'])**2)
            if dist < 5.0:
                duplicates_removed.append(star)
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(star)

    print(f"  Grid detection: {len(candidates)} candidates, {len(deduplicated)} after deduplication")

    if use_starfinder and len(deduplicated) > 0:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            gray_image = image.astype(float)

        refined_stars = []
        for star in deduplicated:
            rx, ry, sharpness, roundness, flux, peak = _centroid_and_metrics(
                gray_image, star['x'], star['y'], refine_window * 2 + 1, ih, iw
            )
            refined_stars.append({
                'x_grid': star['x'], 'y_grid': star['y'],
                'x': rx, 'y': ry,
                'sharpness': sharpness, 'roundness': roundness,
                'flux': flux, 'peak': peak
            })

        n_refined = sum(1 for s in refined_stars if not np.isnan(s['sharpness']))
        print(f"  Centroid refinement: {n_refined}/{len(refined_stars)} stars refined")
        return refined_stars
    else:
        refined_stars = []
        for star in deduplicated:
            refined_stars.append({
                'x_grid': star['x'], 'y_grid': star['y'],
                'x': star['x'], 'y': star['y'],
                'sharpness': np.nan, 'roundness': np.nan,
                'flux': np.nan, 'peak': np.nan
            })
        return refined_stars
