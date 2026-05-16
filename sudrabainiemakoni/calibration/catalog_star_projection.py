import numpy as np
import pandas as pd
import astropy.coordinates
import astropy.units as u
import pymap3d
try:
    from sudrabainiemakoni.calibration.catalog_stars import cat
    from sudrabainiemakoni.calibration.atmospheric_refraction import apply_atmospheric_refraction_correction
except ImportError:
    from catalog_stars import cat
    from atmospheric_refraction import apply_atmospheric_refraction_correction


def project_catalog_stars(ct_camera, location, observation_time,
                          overshoot_px=20, min_magnitude=-3, max_magnitude=4, min_altitude=0.0,
                          skip_pixel_filter=False):

    df_catalog = pd.DataFrame(cat)
    df_catalog.columns = ['ra', 'pmra', 'dec', 'pmdec', 'mag', 'name']

    df_cat = df_catalog[(df_catalog['mag'] <= max_magnitude) & (df_catalog['mag'] > min_magnitude)]
    altaz = astropy.coordinates.AltAz(obstime=observation_time, location=location)

    stars_J2000 = astropy.coordinates.SkyCoord(
        ra=df_cat['ra'].values * u.hour,
        dec=df_cat['dec'].values * u.deg,
        pm_ra_cosdec=df_cat['pmra'].values * u.mas / u.year,
        pm_dec=df_cat['pmdec'].values * u.mas / u.year,
        obstime=astropy.time.Time(2000, format="byear"),
        frame="icrs",
    )

    stars_altaz = stars_J2000.transform_to(altaz)
    valid_min_altitude = stars_altaz.alt > min_altitude * u.deg
    stars_altaz = stars_altaz[valid_min_altitude]

    alt_deg = stars_altaz.alt.deg
    alt_corrected_deg = apply_atmospheric_refraction_correction(alt_deg)

    enu_unit_coords = np.array(pymap3d.aer2enu(stars_altaz.az.value, alt_corrected_deg, 1)).T

    pixel_coords = ct_camera.imageFromSpace(enu_unit_coords, hide_backpoints=True)

    image_width, image_height = ct_camera.image_width_px, ct_camera.image_height_px

    if skip_pixel_filter:
        valid = np.ones(len(pixel_coords), dtype=bool)
    else:
        valid = (pixel_coords[:, 0] >= -overshoot_px) & (pixel_coords[:, 0] < image_width + overshoot_px) & \
                (pixel_coords[:, 1] >= -overshoot_px) & (pixel_coords[:, 1] < image_height + overshoot_px)

    df_stars_in_image = df_cat[valid_min_altitude][valid].copy()
    df_stars_in_image['pixel_x'] = pixel_coords[valid, 0]
    df_stars_in_image['pixel_y'] = pixel_coords[valid, 1]
    df_stars_in_image['altitude'] = stars_altaz.alt.deg[valid]
    df_stars_in_image['azimuth'] = stars_altaz.az.deg[valid]

    # Convert RA from hours to degrees for consistency with the rest of the system
    df_stars_in_image['ra'] = df_stars_in_image['ra'] * 15.0  # 1 hour = 15 degrees

    df_stars_in_image = df_stars_in_image[['name', 'mag', 'ra', 'dec',
                                           'altitude', 'azimuth',
                                           'pixel_x', 'pixel_y']].reset_index(drop=True)

    return df_stars_in_image


def project_catalog_stars_debug(ct_camera, location, observation_time,
                                debug_names,
                                overshoot_px=20, min_magnitude=-3, max_magnitude=4,
                                min_altitude=0.0, skip_pixel_filter=False):
    """Project only the named debug stars, printing each pipeline step."""
    import astropy.time

    if isinstance(debug_names, str):
        debug_names = [debug_names]

    df_catalog = pd.DataFrame(cat)
    df_catalog.columns = ['ra', 'pmra', 'dec', 'pmdec', 'mag', 'name']

    altaz = astropy.coordinates.AltAz(obstime=observation_time, location=location)
    print(f"[debug_proj] location={location}  obstime={observation_time}")

    for name in debug_names:
        print(f"\n[debug_proj] === {name} ===")
        rows = df_catalog[df_catalog['name'] == name]
        if rows.empty:
            print(f"  NOT FOUND in catalog")
            continue

        row = rows.iloc[0]
        ra_h, dec_deg, mag = row['ra'], row['dec'], row['mag']
        print(f"  catalog: ra={ra_h:.6f}h  dec={dec_deg:.6f}deg  mag={mag:.2f}")

        if mag > max_magnitude:
            print(f"  FILTERED by magnitude: mag={mag:.2f} > max_magnitude={max_magnitude}")
        elif mag <= min_magnitude:
            print(f"  FILTERED by magnitude: mag={mag:.2f} <= min_magnitude={min_magnitude}")
        else:
            print(f"  magnitude OK: {min_magnitude} < {mag:.2f} <= {max_magnitude}")

        star_J2000 = astropy.coordinates.SkyCoord(
            ra=ra_h * u.hour,
            dec=dec_deg * u.deg,
            pm_ra_cosdec=row['pmra'] * u.mas / u.year,
            pm_dec=row['pmdec'] * u.mas / u.year,
            obstime=astropy.time.Time(2000, format="byear"),
            frame="icrs",
        )
        star_altaz = star_J2000.transform_to(altaz)
        alt_deg_raw = float(star_altaz.alt.deg)
        az_deg = float(star_altaz.az.deg)
        print(f"  altaz (pre-refraction): alt={alt_deg_raw:.4f}deg  az={az_deg:.4f}deg")

        alt_corrected = float(apply_atmospheric_refraction_correction(np.array([alt_deg_raw]))[0])
        print(f"  alt (post-refraction):  alt={alt_corrected:.4f}deg")

        if alt_corrected <= min_altitude:
            print(f"  FILTERED by altitude: alt={alt_corrected:.4f} <= min_altitude={min_altitude}")
            continue
        else:
            print(f"  altitude OK: {alt_corrected:.4f} > {min_altitude}")

        enu = np.array(pymap3d.aer2enu(az_deg, alt_corrected, 1.0)).reshape(1, 3)
        print(f"  ENU unit vector: {enu[0]}")

        pixel_coords = ct_camera.imageFromSpace(enu, hide_backpoints=True)
        px, py = float(pixel_coords[0, 0]), float(pixel_coords[0, 1])
        print(f"  projected pixel: ({px:.2f}, {py:.2f})")

        w, h = ct_camera.image_width_px, ct_camera.image_height_px
        if skip_pixel_filter:
            print(f"  pixel filter: SKIPPED")
        else:
            in_bounds = (px >= -overshoot_px and px < w + overshoot_px and
                         py >= -overshoot_px and py < h + overshoot_px)
            print(f"  pixel bounds: [{-overshoot_px}, {w+overshoot_px}) x [{-overshoot_px}, {h+overshoot_px})")
            if in_bounds:
                print(f"  pixel filter: IN BOUNDS")
            else:
                print(f"  pixel filter: OUT OF BOUNDS")
