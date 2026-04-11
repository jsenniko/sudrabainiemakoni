"""
StarReference class for managing star reference points in astronomical images.
Supports both RA/DEC coordinates (from star catalogs) and Alt-Az coordinates (direct measurements).
"""

import numpy as np
import astropy.coordinates
import astropy.units as u
import pymap3d


def apply_atmospheric_refraction_correction(alt_deg):
    """
    Apply atmospheric refraction correction to altitude angle.

    This correction accounts for the bending of light through Earth's atmosphere,
    which causes celestial objects to appear slightly higher than their true position.
    The formula is based on Bennett's empirical refraction formula.

    Args:
        alt_deg: Altitude angle(s) in degrees (scalar or array)

    Returns:
        Corrected altitude angle(s) in degrees
    """
    return alt_deg + 0.01666 / np.tan(np.radians(alt_deg + (7.31 / (alt_deg + 4.4))))


class StarReference:
    """
    A reference point representing a star in an astronomical image.
    
    Can store either:
    - RA/DEC coordinates (resolved from star names or direct coordinates)
    - Alt-Az coordinates (direct measurements from the image)
    
    Maintains backward compatibility with existing pickle files.
    """
    
    def __init__(self, name, pixelcoords, altaz_coord=None):
        """
        Initialize a star reference with automatic coordinate parsing.
        
        The name parameter supports multiple formats:
        - Star name: "Sirius", "Polaris"
        - Alt-Az coordinates: "120.5,45.2" (azimuth,altitude in degrees)
        - RA/DEC coordinates: "ra:83.633,22.014" (degrees)
        
        Args:
            name: Star name or coordinate string
            pixelcoords: [x, y] pixel coordinates in image
            altaz_coord: Optional astropy.coordinates.AltAz object for direct Alt-Az coordinates
        """
        self.pixelcoords = pixelcoords
        self.skycoord: astropy.coordinates.SkyCoord = None
        
        # Parse the name for different coordinate formats
        parsed_name, coord_type, parsed_coords = self._parse_name_input(name)
        self.name = parsed_name
        
        # Handle parsed coordinates
        if coord_type == 'altaz' and parsed_coords is not None:
            az, alt = parsed_coords
            self.altaz_coord = astropy.coordinates.AltAz(
                az=az * u.deg,
                alt=alt * u.deg
            )
        elif coord_type == 'radec' and parsed_coords is not None:
            ra, dec = parsed_coords
            self.skycoord = astropy.coordinates.SkyCoord(ra, dec, unit='deg')
            self.altaz_coord = None
        else:
            # Traditional name or explicit altaz_coord parameter
            if not hasattr(self, 'altaz_coord'):
                self.altaz_coord: astropy.coordinates.AltAz = altaz_coord
    
    def _parse_name_input(self, name_input):
        """
        Parse name input for different coordinate formats.
        
        Returns:
            tuple: (display_name, coordinate_type, parsed_coords)
        """
        if not isinstance(name_input, str):
            return str(name_input), 'name', None
            
        name_input = name_input.strip()
        
        # Check for RA/DEC format: ra:XXX,YYY
        if name_input.lower().startswith('ra:'):
            try:
                coords_part = name_input[3:].strip()  # Remove 'ra:' prefix
                parts = coords_part.split(',')
                
                if len(parts) != 2:
                    print("RA/DEC formāts: ra:xxx,yyy")
                    return name_input, 'invalid', None
                    
                ra = float(parts[0].strip())
                dec = float(parts[1].strip())
                
                # Validate ranges
                if not (0 <= ra <= 360):
                    print(f"Rektascensija ārpus diapazona 0-360°: {ra}")
                    return name_input, 'invalid', None
                if not (-90 <= dec <= 90):
                    print(f"Deklinācija ārpus diapazona -90 līdz 90°: {dec}")
                    return name_input, 'invalid', None
                
                # Create display name
                display_name = f"RA/DEC({ra:.2f}°,{dec:.2f}°)"
                return display_name, 'radec', (ra, dec)
                
            except (ValueError, IndexError) as e:
                print(f"Kļūda RA/DEC koordinātu parsēšanā: {e}")
                return name_input, 'invalid', None
        
        # Check for Alt-Az format: XXX.xxx,YYY.yyy (simple comma-separated numbers)
        elif ',' in name_input:
            try:
                parts = name_input.split(',')
                if len(parts) == 2:
                    az = float(parts[0].strip())
                    alt = float(parts[1].strip())
                    
                    # Validate ranges
                    if not (0 <= az <= 360):
                        print(f"Azimuts ārpus diapazona 0-360°: {az}")
                        return name_input, 'invalid', None
                    if not (-90 <= alt <= 90):
                        print(f"Augstums ārpus diapazona -90 līdz 90°: {alt}")
                        return name_input, 'invalid', None
                    
                    # Create display name
                    display_name = f"Az/Alt({az:.1f}°,{alt:.1f}°)"
                    return display_name, 'altaz', (az, alt)
            except ValueError as e:
                print(f"Kļūda Alt-Az koordinātu parsēšanā: {e}")
                # Fall through to treat as star name
        
        # Default: treat as star name
        return name_input, 'name', None
    
    def __str__(self):
        return f"{self.name} {self.pixelcoords}"
    
    def __repr__(self):
        if hasattr(self, 'altaz_coord') and self.altaz_coord is not None:
            return f"{self.name} AltAz({self.altaz_coord.az.deg:.3f}°, {self.altaz_coord.alt.deg:.3f}°)"
        elif self.skycoord is not None:
            return f"{self.name} {self.skycoord.__repr__()}"
        else:
            return f"{self.name} (unresolved)"
    
    def getSkyCoord(self):
        """
        Get RA/DEC coordinates for this star.
        Original method - maintains backward compatibility.
        """
        # If we already have a skycoord, return it
        if self.skycoord is not None:
            return self.skycoord
            
        # If we have Alt-Az coordinates, we can't convert to RA/DEC without a specific time and location
        # This should be handled by the CloudImage class that has access to the observation context
        if hasattr(self, 'altaz_coord') and self.altaz_coord is not None:
            return None
            
        # Original logic for resolving from name
        try:
            # literal ra, dec in degrees
            l = self.name.split(',')
            if len(l) == 2:
                ra, dec = float(l[0]), float(l[1])
                c = astropy.coordinates.SkyCoord(ra, dec, unit='deg')
            else:
                # Try to resolve star name
                c = astropy.coordinates.SkyCoord.from_name(self.name)
        except Exception as e:
            c = astropy.coordinates.SkyCoord.from_name(self.name)
            
        print(self.name, c)
        self.skycoord = c
        return self.skycoord
    
    def getAltAzCoord(self, altaz_frame=None):
        """
        Get Alt-Az coordinates for this star.
        
        Args:
            altaz_frame: astropy.coordinates.AltAz frame for coordinate transformation
            
        Returns:
            astropy.coordinates.AltAz or None if not available
        """
        # If we have direct Alt-Az coordinates, return them
        if hasattr(self, 'altaz_coord') and self.altaz_coord is not None:
            return self.altaz_coord
            
        # If we have RA/DEC and a frame, convert to Alt-Az
        if self.skycoord is not None and altaz_frame is not None:
            return self.skycoord.transform_to(altaz_frame)
            
        return None
    
    def hasDirectAltAz(self):
        """Check if this star has direct Alt-Az coordinates (not derived from RA/DEC)"""
        return hasattr(self, 'altaz_coord') and self.altaz_coord is not None

    def hasDirectRADEC(self):
        """Check if this star has RA/DEC coordinates (skycoord)"""
        return self.skycoord is not None
    
    def setAltAzCoord(self, az_deg, alt_deg):
        """Set Alt-Az coordinates directly"""
        self.altaz_coord = astropy.coordinates.AltAz(
            az=az_deg * u.deg,
            alt=alt_deg * u.deg
        )
    
    def getENUUnitVector(self, altaz_frame=None, refraction_correction=True):
        """
        Get ENU unit vector for this star.
        This is used by get_stars_enu_unit_coords.

        Args:
            altaz_frame: astropy.coordinates.AltAz frame (needed if converting from RA/DEC)
            refraction_correction: Apply atmospheric refraction correction (default True)

        Returns:
            numpy array [east, north, up] or None if coordinates not available
        """
        altaz_coord = self.getAltAzCoord(altaz_frame)
        if altaz_coord is not None:
            alt_deg = altaz_coord.alt.value

            # Apply atmospheric refraction correction if requested
            if refraction_correction:
                alt_deg = apply_atmospheric_refraction_correction(alt_deg)

            enu = pymap3d.aer2enu(altaz_coord.az.value, alt_deg, 1)
            return np.array(enu)
        return None


def get_stars_enu_unit_coords(star_references: list[StarReference], altaz_frame, refraction_correction=True):
    """
    Extract ENU unit vectors from a list of star references.

    Args:
        star_references: List of StarReference objects
        altaz_frame: astropy.coordinates.AltAz frame for coordinate transformation
        refraction_correction: Apply atmospheric refraction correction (default True)

    Returns:
        numpy array of shape (n_stars, 3) with ENU unit vectors, or empty array if no valid stars
    """
    enu_coords = []

    for star in star_references:
        # Try to get ENU coordinates from each star
        enu = star.getENUUnitVector(altaz_frame, refraction_correction=refraction_correction)
        if enu is not None:
            enu_coords.append(enu)
        else:
            print(f'WARNING: star without coordinates {star}')

    if enu_coords:
        return np.array(enu_coords)
    else:
        return np.array([])

