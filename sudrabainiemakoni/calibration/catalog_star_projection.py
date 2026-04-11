import numpy as np
import pandas as pd
import astropy.coordinates
import astropy.units as u
import pymap3d
from sudrabainiemakoni.calibration.catalog_stars import cat
from sudrabainiemakoni.starreference import apply_atmospheric_refraction_correction


def project_catalog_stars(camera, location, observation_time,
                          overshoot_px=20, min_magnitude=-3, max_magnitude=4, min_altitude=0.0):

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
    valid_min_altitude = stars_altaz.alt > min_altitude
    stars_altaz = stars_altaz[valid_min_altitude]

    alt_deg = stars_altaz.alt.deg
    alt_corrected_deg = apply_atmospheric_refraction_correction(alt_deg)

    enu_unit_coords = np.array(pymap3d.aer2enu(stars_altaz.az.value, alt_corrected_deg, 1)).T

    pixel_coords = camera.camera_enu.imageFromSpace(enu_unit_coords, hide_backpoints=True)

    image_width, image_height = camera.image_size

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
