__author__ = 'Juris Seņņikovs'
import astropy
import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np
from sudrabainiemakoni import calculations

class AAZImage:
    def __init__(self, cloudImage):
        self.cloudImage = cloudImage
    def Initialize(self,  width=None,  height=None):
        x=self.cloudImage.aazgrid.az.wrap_at(180*u.deg).value
        y=self.cloudImage.aazgrid.alt.value
        self.azmin, self.azmax, self.altmin, self.altmax = np.min(x),np.max(x), np.min(y), np.max(y)
        # prepare image transformed to alt az coordinates and corresponding reference information
        if width is None:
            width = self.cloudImage.imagearray.shape[1]
        if height is None:
            height = self.cloudImage.imagearray.shape[0]
        self.aaz_image=np.zeros(shape=(height, width,3), dtype='uint8')
        az_arr=np.linspace(self.azmin,self.azmax, self.aaz_image.shape[1])
        alt_arr=np.linspace(self.altmax,self.altmin, self.aaz_image.shape[0])
        self.az_grid, self.alt_grid = np.meshgrid(az_arr, alt_arr)
    def Prepare_throughRA(self):
        # veidojam augstuma/azimuta koodinātēs veidotu bitmapu
        # tas vajadzīgs, lai būtu iespējama transformācija no piemēram lat/lon/height mākoņos uz AltAz un varētu izvēlēties pikseli
        # šajā apakšprogrammāto darām caur rektascensiju un deklināciju (jo pagaidām nemāku izveidot WCS objektu AltAz koordinātēs)
        # vēl iespējamā metode ir fitēt kamera leņķus, attēla centru (ja nu gadījumā ir apgriezts) un skata leņķi
        # vēl iespējams arī inteprolēt pēc tiešās piešķiršanas piem ar scipy.ndimage.spline_filter
        print('Prepare AAZImage grid through RA')
        cc_grid = astropy.coordinates.SkyCoord(astropy.coordinates.AltAz(
                location = self.cloudImage.aazgrid.location, obstime = self.cloudImage.aazgrid.obstime, az=self.az_grid*u.deg, alt=self.alt_grid*u.deg))
        ra_dec_grid=cc_grid.transform_to(astropy.coordinates.ICRS())
        # find map of pixels from original_image to aazimage
        self.aaz_pixels_in_original_image = self.cloudImage.wcs.world_to_pixel(ra_dec_grid)
        self.aaz_pixels_in_original_image = np.array(self.aaz_pixels_in_original_image[1], dtype='int'), np.array(self.aaz_pixels_in_original_image[0], dtype='int')
        self.valid = (self.aaz_pixels_in_original_image[0]>=0) & (self.aaz_pixels_in_original_image[0]<self.cloudImage.imagearray.shape[0]) \
                & (self.aaz_pixels_in_original_image[1]>=0) & (self.aaz_pixels_in_original_image[1]<self.cloudImage.imagearray.shape[1])
        self.iaz_grid, self.ialt_grid = np.meshgrid(np.arange(0, self.aaz_image.shape[1]), np.arange(0, self.aaz_image.shape[0]))
    def FillImage(self):
        self.aaz_image[self.ialt_grid[self.valid], self.iaz_grid[self.valid]]= \
            self.cloudImage.imagearray[self.aaz_pixels_in_original_image[0][self.valid],self.aaz_pixels_in_original_image[1][self.valid]]
    def GetPixelCoords(self, az, alt):
        azpix=np.round((az-self.azmin)/(self.azmax-self.azmin)*self.aaz_image.shape[1]).astype('int')
        altpix=np.round((self.altmax - alt)/(self.altmax-self.altmin)*self.aaz_image.shape[0]).astype('int')

        return azpix, altpix

class LatLonImage:
    def __init__(self, cloudImage, lonmin, lonmax, latmin, latmax, npix_lon, npix_lat):
        self.lonmin, self.lonmax, self.latmin, self.latmax =  lonmin, lonmax, latmin, latmax
        self.npix_lon, self.npix_lat = npix_lon, npix_lat
        self.longitudes = np.linspace(lonmin,lonmax, npix_lon)
        self.latitudes = np.linspace(latmin,latmax, npix_lat)
        self.lon_grid, self.lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        self.ilon_grid, self.ilat_grid = np.meshgrid(np.arange(0, npix_lon), np.arange(0, npix_lat))
        self.cloudImage = cloudImage

    def prepare_reproject(self, height_km):
        height = height_km* u.km
        location_grid = astropy.coordinates.EarthLocation(lon = self.lon_grid, lat=self.lat_grid, height = height)
        itrs_grid = astropy.coordinates.ITRS(x=location_grid.x, y=location_grid.y, z=location_grid.z, obstime=self.cloudImage.date)
        # šeit sanāks alt/az kopā ar attālumu no novērotāja
        aaz_grid = itrs_grid.transform_to(self.cloudImage.altaz)
        aaz_grid =astropy.coordinates.AltAz(location = aaz_grid.location, obstime = aaz_grid.obstime, az=aaz_grid.az, alt=aaz_grid.alt)
        print('Got AAZ_GRID')
        # find pixels in AltAz image
        # eliminate circular jump at north
        x=aaz_grid.az.wrap_at(180*u.deg).value
        y=aaz_grid.alt.value
        aazimage = self.cloudImage.AAZImage
        self.azpix, self.altpix = aazimage.GetPixelCoords(x, y)
        self.maskpix=(self.azpix>=0) & (self.azpix<aazimage.aaz_image.shape[1]) & (self.altpix>=0) & (self.altpix<aazimage.aaz_image.shape[0])

    def Fill_projectedImage(self):
        projected_image=np.zeros(shape=(self.npix_lat, self.npix_lon, 3), dtype='uint8')
        aazimage = self.cloudImage.AAZImage
        projected_image[self.ilat_grid[self.maskpix], self.ilon_grid[self.maskpix]]=aazimage.aaz_image[self.altpix[self.maskpix],self.azpix[self.maskpix]]
        return projected_image
    def GetPixelCoords(self, longitudes, latitudes):
        longitude_pix = (longitudes-self.lonmin)/(self.lonmax-self.lonmin)*self.npix_lon
        latitude_pix = (latitudes-self.latmin)/(self.latmax-self.latmin)*self.npix_lat
        return longitude_pix, latitude_pix
    def GetJgw(self):
        return [(self.lonmax-self.lonmin)/self.npix_lon,
                0,
                0,
                (self.latmax-self.latmin)/self.npix_lat,
                self.lonmin,
               self.latmin,]
    def SaveJgw(self, filename):
        jgw = self.GetJgw()
        with open(filename,'w') as f:
            for l in jgw:
                f.write(str(l)+'\n')


class WCSCoordinateSystemsAdapter:
    """
    Adapter class to provide WCS-based coordinate system functionality 
    for CloudImage objects that have been refactored to use camera-based systems.
    """
    
    def __init__(self, cloud_image):
        self.cloud_image = cloud_image
        self._wcs = None
        self._radecgrid = None
        self._aazgrid = None
        self._AAZImage = None
    
    def GetWCS(self, sip_degree=2, fit_parameters={}):
        """
        Create WCS (World Coordinate System) object from star references.
        """
        # wcs objekta atrašana no zvaigznēm atbilsotšajām pikseļu koordinātēm
        pixelcoords = self.cloud_image.getPixelCoords()
        skycoords = self.cloud_image.getSkyCoords()
        wcs = astropy.wcs.utils.fit_wcs_from_points((pixelcoords[:,0], pixelcoords[:,1]),
                astropy.coordinates.SkyCoord(skycoords), sip_degree=sip_degree, **fit_parameters)
        print(wcs)
        self._wcs = wcs
        return wcs
    
    @property
    def wcs(self):
        if self._wcs is None:
            self.GetWCS()
        return self._wcs
    
    def TestStarFit(self):
        """Test WCS fit quality against star references."""
        wcs = self.wcs
        for starref in self.cloud_image.starReferences:
            px = starref.pixelcoords
            sc = wcs.pixel_to_world(*px)
            x, y = wcs.world_to_pixel(starref.skycoord)
            dist = np.sqrt((px[0]-x)**2+(px[1]-y)**2)
            print(starref.name, sc.separation(starref.skycoord), dist)
    
    @property
    def radecgrid(self):
        if self._radecgrid is None:
            self.GetImageRaDecGrid()
        return self._radecgrid
    
    def GetImageRaDecGrid(self, reload=False):
        if self._radecgrid is None or reload:
            print('Calculate RaDec grid')
            self._radecgrid = calculations.GetImageRaDecGrid(self.cloud_image.imagearray, self.wcs)
    
    @property
    def aazgrid(self):
        if self._aazgrid is None:
            self.GetAltAzGrid()
        return self._aazgrid
    
    def GetAltAzGrid(self, reload=False):
        if self._aazgrid is None or reload:
            print('Calculate AltAz grid')
            self._aazgrid = self.radecgrid.transform_to(self.cloud_image.altaz)
    
    @property
    def AAZImage(self):
        if self._AAZImage is None:
            self.Initialize_AltAzImage()
        return self._AAZImage
    
    def Initialize_AltAzImage(self, width=None, height=None, reload=False):
        if self._AAZImage is None or reload:
            # Create a temporary cloud image wrapper that has the required properties
            cloud_image_wrapper = WCSCloudImageWrapper(self.cloud_image, self)
            self._AAZImage = AAZImage(cloud_image_wrapper)
            self._AAZImage.Initialize(width, height)
    
    def PrepareAltAZImage_throughRA(self):
        self.AAZImage.Prepare_throughRA()
    
    def GetAAzCoord(self, pixelcoords):
        """Get AltAz coordinates for given pixel coordinates."""
        wcs = self.wcs
        skycoord = wcs.pixel_to_world(*pixelcoords)
        aazc = skycoord.transform_to(self.cloud_image.altaz)
        return aazc


class WCSCloudImageWrapper:
    """
    Wrapper class to make a modern CloudImage compatible with legacy WCS-based classes
    like AAZImage and LatLonImage.
    """
    
    def __init__(self, cloud_image, wcs_adapter):
        self.cloud_image = cloud_image
        self.wcs_adapter = wcs_adapter
    
    @property
    def imagearray(self):
        return self.cloud_image.imagearray
    
    @property
    def wcs(self):
        return self.wcs_adapter.wcs
    
    @property
    def aazgrid(self):
        return self.wcs_adapter.aazgrid
    
    @property
    def date(self):
        return self.cloud_image.date
    
    @property
    def altaz(self):
        return self.cloud_image.altaz
    
    @property
    def AAZImage(self):
        return self.wcs_adapter.AAZImage