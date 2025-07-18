"""
Web Mercator projection class derived from ProjectionImagePyproj.
This class provides Web Mercator (EPSG:3857) projection functionality
while inheriting all the common projection features from the base classes.
"""

from .projection import ProjectionImagePyproj


class ProjectionImageWebMercator(ProjectionImagePyproj):
    """
    Web Mercator projection implementation.
    
    This class automatically sets up the Web Mercator projection (EPSG:3857)
    and provides all the functionality needed for projecting cloud images
    to Web Mercator coordinates.
    """
    
    def __init__(self, cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km):
        """
        Initialize Web Mercator projection.
        
        Args:
            cloudImage: CloudImage instance
            lonmin: Minimum longitude
            lonmax: Maximum longitude  
            latmin: Minimum latitude
            latmax: Maximum latitude
            pixel_per_km: Resolution in pixels per kilometer
        """
        # Set up Web Mercator projection automatically
        super().__init__("EPSG:3857", cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km)