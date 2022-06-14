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
