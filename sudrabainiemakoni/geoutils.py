import numpy as np
#modified from https://stephenhartzell.medium.com/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
def los_to_constant_height_surface(x,y,z, u,v,w, surface_height):
    """
    Finds the intersection of a pointing vector u and starting point s with the constant height surface above WGS-84 ellipsoid
    Args:
        position  (x,y,z)):  the starting point location(s) in meters (ECEF)
        pointing (u,v,w): length 3 array defining the pointing vector(s) (must be a unit vector)
    Returns:
        np.array: length 3 defining the point(s) of intersection with the surface of the Earth in meters
    """
    aa=6378137.0+surface_height
    bb= 6356752.314245+surface_height
    a = aa
    b = aa
    c = bb
    #x = position[0]
    #y = position[1]
    #z = position[2]
    #u = pointing[0]
    #v = pointing[1]
    #w = pointing[2]

    value = -a**2*b**2*w*z - a**2*c**2*v*y - b**2*c**2*u*x
    radical = a**2*b**2*w**2 + a**2*c**2*v**2 - a**2*v**2*z**2 + 2*a**2*v*w*y*z - a**2*w**2*y**2 + b**2*c**2*u**2 - b**2*u**2*z**2 + 2*b**2*u*w*x*z - b**2*w**2*x**2 - c**2*u**2*y**2 + 2*c**2*u*v*x*y - c**2*v**2*x**2
    magnitude = a**2*b**2*w**2 + a**2*c**2*v**2 + b**2*c**2*u**2


    radical=np.ma.masked_array(radical, mask=radical<0)
    #if radical < 0:
    #    raise ValueError("The Line-of-Sight vector does not point from the Earth")

    d = (value + a*b*c*np.sqrt(radical)) / magnitude

    #if d <= 0:
    #    raise ValueError(f"The Line-of-Sight vector does not point from the Earth {d}")
    d=np.ma.masked_array(d, mask=d<0)
    return np.array([
        x + d * u,
        y + d * v,
        z + d * w,
    ])


def los_to_earth(x,y,z, u,v,w, surface_height, to_earth = True):
    """
    Finds the intersection of a pointing vector u and starting point s with the constant height surface above WGS-84 ellipsoid
    Args:
        position  (x,y,z)):  the starting point location(s) in meters (ECEF)
        pointing (u,v,w): length 3 array defining the pointing vector(s) (must be a unit vector)
        to_earth - direction to consider True - look to earth, False - look from earth
    Returns:
        np.array: length 3 defining the point(s) of intersection with the surface of the Earth in meters
    """
    aa=6378137.0+surface_height
    bb= 6356752.314245+surface_height
    a = aa
    b = aa
    c = bb
    value = -(a ** 2) * b ** 2 * w * z - a ** 2 * c ** 2 * v * y - b ** 2 * c ** 2 * u * x
    radical = (
        a ** 2 * b ** 2 * w ** 2
        + a ** 2 * c ** 2 * v ** 2
        - a ** 2 * v ** 2 * z ** 2
        + 2 * a ** 2 * v * w * y * z
        - a ** 2 * w ** 2 * y ** 2
        + b ** 2 * c ** 2 * u ** 2
        - b ** 2 * u ** 2 * z ** 2
        + 2 * b ** 2 * u * w * x * z
        - b ** 2 * w ** 2 * x ** 2
        - c ** 2 * u ** 2 * y ** 2
        + 2 * c ** 2 * u * v * x * y
        - c ** 2 * v ** 2 * x ** 2
    )

    magnitude = a ** 2 * b ** 2 * w ** 2 + a ** 2 * c ** 2 * v ** 2 + b ** 2 * c ** 2 * u ** 2

    # %%   Return nan if radical < 0 or d < 0 because LOS vector does not point towards Earth
    try:
        if to_earth:
            d = (value - a * b * c * np.sqrt(radical)) / magnitude
        else:
            d = (value + a * b * c * np.sqrt(radical)) / magnitude
        d[radical < 0] = np.nan
        d[d < 0] = np.nan
    except ValueError:
        if radical < 0:
            d = np.nan
        if d < 0:
            d = np.nan
    except TypeError:
        pass
    return np.array([
        x + d * u,
        y + d * v,
        z + d * w,
    ])

def get_is_sunlit(x,y,z, astropy_date, atmosphere_width_km=0.0):
    """
    x,y,z -  ECEF
    """
    import astropy
    import astropy.units as u

    sun_pos = astropy.coordinates.get_sun(astropy_date)
    itrs=astropy.coordinates.ITRS(obstime=astropy_date)
    pp=sun_pos.transform_to(itrs)
    sun_xyz = pp.cartesian.xyz.to(u.m).value
    vector_to_sun = sun_xyz-np.stack([x,y,z]).T
    vector_to_sun = vector_to_sun/np.linalg.norm(vector_to_sun, axis=-1)[...,np.newaxis]
    xxx=los_to_earth(x.flatten(),y.flatten(),z.flatten(),
             vector_to_sun[...,0].flatten(),vector_to_sun[...,1].flatten(),vector_to_sun[...,2].flatten(), 
             atmosphere_width_km*1000.0,
                             to_earth=True)
    is_sunlit=np.isnan(xxx[0])
    is_sunlit=is_sunlit.reshape(x.shape)
    return is_sunlit
def get_is_sunlit_latlon(lat,lon,height, astropy_date,atmosphere_width_km=0.0):
    import pymap3d
    x,y,z=pymap3d.geodetic2ecef(lat,lon, height)
    return get_is_sunlit(x,y,z,astropy_date,atmosphere_width_km=atmosphere_width_km)