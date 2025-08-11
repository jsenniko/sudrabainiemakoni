"""
StarReference class for managing star reference points in astronomical images.
Supports both RA/DEC coordinates (from star catalogs) and Alt-Az coordinates (direct measurements).
"""

import numpy as np
import astropy.coordinates
import astropy.units as u
import pymap3d


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
    
    def setAltAzCoord(self, az_deg, alt_deg):
        """Set Alt-Az coordinates directly"""
        self.altaz_coord = astropy.coordinates.AltAz(
            az=az_deg * u.deg,
            alt=alt_deg * u.deg
        )
    
    def getENUUnitVector(self, altaz_frame=None):
        """
        Get ENU unit vector for this star.
        This is used by get_stars_enu_unit_coords.
        
        Args:
            altaz_frame: astropy.coordinates.AltAz frame (needed if converting from RA/DEC)
            
        Returns:
            numpy array [east, north, up] or None if coordinates not available
        """
        altaz_coord = self.getAltAzCoord(altaz_frame)
        if altaz_coord is not None:
            enu = pymap3d.aer2enu(altaz_coord.az.value, altaz_coord.alt.value, 1)
            return np.array(enu)
        return None

