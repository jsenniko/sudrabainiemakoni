__author__ = 'Juris Seņņikovs'
import os
import astropy
import astropy.coordinates
import astropy.units as u
import astropy.wcs
import datetime, pytz
import numpy as np
import pandas as pd
import skimage
import skimage.io
import cameratransform as ct
import pymap3d
from scipy.spatial.transform import Rotation
import scipy.optimize
import utils
import geoutils
import optimize_camera
import calculations
class StarReference:
    def __init__(self, name, pixelcoords):
        self.name = name
        self.pixelcoords = pixelcoords
        self.skycoord: astropy.coordinates.SkyCoord = None
    def __str__(self):
        return f"{self.name}"
    def __repr__(self):
        return f"{self.name} {self.skycoord.__repr__()}"
    def getSkyCoord(self):
        c=astropy.coordinates.SkyCoord.from_name(self.name)
        print(self.name, c)
        self.skycoord = c

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

class WebMercatorImage:
    import pyproj
    TRAN_4326_TO_3857 = pyproj. Transformer.from_crs("EPSG:4326", "EPSG:3857")
    TRAN_3857_TO_4326 = pyproj. Transformer.from_crs("EPSG:3857","EPSG:4326")

    def __init__(self, cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km):
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax
        self.pixel_per_km = pixel_per_km
        bounds = [(lonmin,latmin),(lonmin,latmax),(lonmax,latmin),(lonmax,latmax)]
        x, y = WebMercatorImage.TRAN_4326_TO_3857.transform([b[1] for b in bounds],[b[0] for b in bounds])
        self.xmin = min(x)
        self.xmax = max(x)
        self.ymin = min(y)
        self.ymax = max(y)
        self.npix_x, self.npix_y = int((self.xmax-self.xmin)/(pixel_per_km*1000)),int((self.ymax-self.ymin)/(pixel_per_km*1000))
        self.cloudImage = cloudImage
        self.initialize()

    def initialize(self):
        self.x_arr = np.linspace(self.xmin,self.xmax, self.npix_x)
        self.y_arr = np.linspace(self.ymax,self.ymin, self.npix_y)
        self.x_grid, self.y_grid = np.meshgrid(self.x_arr, self.y_arr)
        self.ix_grid, self.iy_grid = np.meshgrid(np.arange(0, self.npix_x), np.arange(0, self.npix_y))
        self.lat_grid, self.lon_grid = WebMercatorImage.TRAN_3857_TO_4326.transform(self.x_grid.flatten(),self.y_grid.flatten())
        self.lon_grid = self.lon_grid.reshape(self.x_grid.shape)
        self.lat_grid = self.lat_grid.reshape(self.x_grid.shape)
    fields = {"lonmin","lonmax","latmin","latmax","lonmin","pixel_per_km","xmin", "xmax", "ymin", "ymax", "npix_x", "npix_y"}
    def __getstate__(self):
        state = self.__dict__
        return {x: state[x] for x in WebMercatorImage.fields}
    def __setstate__(self, state):
        for x in WebMercatorImage.fields:
            setattr(self, x, state[x])
        self.initialize()

    def __str__(self):
        return {k:v for k, v in self.__dict__.items() if k in WebMercatorImage.fields}.__str__()
    def __repr__(self):
        return {k:v for k, v in self.__dict__.items() if k in WebMercatorImage.fields}.__repr__()

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
        azimuth=aaz_grid.az.wrap_at(180*u.deg).value
        altitude=aaz_grid.alt.value
        aazimage = self.cloudImage.AAZImage
        self.azpix, self.altpix = aazimage.GetPixelCoords(azimuth, altitude)
        self.maskpix=(self.azpix>=0) & (self.azpix<aazimage.aaz_image.shape[1]) & (self.altpix>=0) & (self.altpix<aazimage.aaz_image.shape[0])

    def Fill_projectedImage(self, from_camera = True):
        projected_image=np.zeros(shape=(self.npix_y, self.npix_x, 3), dtype='uint8')
        if from_camera:
            projected_image[self.iy_grid[self.maskpix], self.ix_grid[self.maskpix]]=self.cloudImage.imagearray[self.j_pxls[self.maskpix], self.i_pxls[self.maskpix]]
        else:
            aazimage = self.cloudImage.AAZImage
            projected_image[self.iy_grid[self.maskpix], self.ix_grid[self.maskpix]]=aazimage.aaz_image[self.altpix[self.maskpix],self.azpix[self.maskpix]]
        return projected_image
    def Fill_projectedImageMasked(self, from_camera = True):
        img = self.Fill_projectedImage(from_camera)
        alpha = ((1-((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0)))*255).astype('uint8')
        masked_img = np.append(img, alpha[:,:,np.newaxis], axis=2)
        return masked_img

    def prepare_reproject_from_camera(self, height_km):
        height = height_km * 1000
        x,y,z = pymap3d.geodetic2ecef(self.lat_grid, self.lon_grid, height)
        xyz = np.array([x.flatten(),y.flatten(),z.flatten()]).T
        image_pxls = self.cloudImage.camera.camera_ecef.imageFromSpace(xyz, hide_backpoints=False)
        image_pxls = np.reshape(image_pxls, (x.shape[0],x.shape[1],2))
        self.i_pxls, self.j_pxls = image_pxls[:,:,0],image_pxls[:,:,1]
        width, height = self.cloudImage.imagearray.shape[1], self.cloudImage.imagearray.shape[0]
        self.maskpix = ~np.isnan(self.i_pxls) & ~np.isnan(self.j_pxls)
        self.i_pxls=np.round(self.i_pxls).astype('int')
        self.j_pxls=np.round(self.j_pxls).astype('int')
        self.maskpix = self.maskpix &  (self.i_pxls>=0) &  (self.i_pxls<width)  & (self.j_pxls>=0) & (self.j_pxls<height)

    def prepare_inverse_reproject_from_camera(self, height_km):
        i_grid, j_grid = self.cloudImage.imageArrayGrid()
        grid_points=np.array([i_grid.flatten(), j_grid.flatten()]).T
        cam = self.cloudImage.camera.camera_ecef
        center, rays = cam.getRay(grid_points, normed=True)
        ray_coords=rays.T
        xyz = geoutils.los_to_constant_height_surface(*center,*ray_coords, height_km*1000)
        lat, lon, h = pymap3d.ecef2geodetic(*xyz)
        i_pxls, j_pxls = self.GetPixelCoords_LatLon(lat, lon)
        i_pxls = np.reshape(i_pxls, (i_grid.shape[0],i_grid.shape[1]))
        j_pxls = np.reshape(j_pxls, (i_grid.shape[0],i_grid.shape[1]))
        maskpix = ~np.isnan(i_pxls) & ~np.isnan(j_pxls)
        i_pxls=np.round(i_pxls).astype('int')
        j_pxls=np.round(j_pxls).astype('int')
        maskpix = maskpix &  (i_pxls>=0) &  (i_pxls<self.npix_x)  & (j_pxls>=0) & (j_pxls<self.npix_y)
        self.inv_i_pxls = i_pxls
        self.inv_j_pxls = j_pxls
        self.inv_maskpix = maskpix
    def Fill_inverse_projected_image(self, projected_image):
        i_grid, j_grid = self.cloudImage.imageArrayGrid()
        if len(projected_image.shape)==2:
            inverse_projected_image=np.zeros(shape=(self.cloudImage.imagearray.shape[0], self.cloudImage.imagearray.shape[1]), dtype=projected_image.dtype)
        else:
            inverse_projected_image=np.zeros(shape=(self.cloudImage.imagearray.shape[0], self.cloudImage.imagearray.shape[1], projected_image.shape[2]), dtype=projected_image.dtype)
        inverse_projected_image[j_grid[self.inv_maskpix], i_grid[self.inv_maskpix]]=projected_image[self.inv_j_pxls[self.inv_maskpix], self.inv_i_pxls[self.inv_maskpix]]
        return inverse_projected_image



    def GetPixelCoords(self, x, y):
        x_pix = (x-self.xmin)/(self.xmax-self.xmin)*self.npix_x
        y_pix = (self.ymax-y)/(self.ymax-self.ymin)*self.npix_y
        return x_pix, y_pix
    def GetPixelCoords_LatLon(self, lat, lon):
        x, y = WebMercatorImage.TRAN_4326_TO_3857.transform(lat, lon)
        x_pix = (x-self.xmin)/(self.xmax-self.xmin)*self.npix_x
        y_pix = (self.ymax-y)/(self.ymax-self.ymin)*self.npix_y
        return x_pix, y_pix
    def GetJgw(self):
        return [(self.xmax-self.xmin)/self.npix_x,
                0,
                0,
                -(self.ymax-self.ymin)/self.npix_y,
                self.xmin,
               self.ymax,]

    def SaveJgw(self, filename):
        jgw = self.GetJgw()
        with open(filename,'w') as f:
            for l in jgw:
                f.write(str(l)+'\n')
    def SaveImageJpg(self, filename):
        img = self.Fill_projectedImage()
        skimage.io.imsave(filename, img)
    def SaveImageTif(self, filename):
        img = self.Fill_projectedImageMasked()
        skimage.io.imsave(filename, img)

    def save(self, filename):
        import pickle

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def load(cls, filename, cloudImage = None):
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        d.cloudImage = cloudImage
        return d
    

    def PrepareHeightMap(self, point_longitudes, point_latitudes, point_heights):
        import pykrige
        assert(len(point_longitudes==len(point_longitudes)))
        assert(len(point_longitudes==len(point_heights)))
        if len(point_longitudes)>=3:
            x, y = WebMercatorImage.TRAN_4326_TO_3857.transform(point_latitudes, point_longitudes)
            OK = pykrige.ok.OrdinaryKriging(
                x, y, point_heights,
                variogram_model="linear",
                verbose=False,
                enable_plotting=False,
                variogram_parameters ={'slope':1.0, 'nugget': 0.0}
            )
            window=10
            krigx, krigy = np.append(self.x_arr[::window],self.x_arr[-1]), np.append(self.y_arr[::window],self.y_arr[-1])
            krigx_grid, krigy_grid = np.meshgrid(krigx, krigy)
            z, ss = OK.execute("grid", krigx, krigy)
            import scipy.interpolate
            heightgrid=scipy.interpolate.griddata((krigx_grid.flatten(), krigy_grid.flatten() ), z.flatten(),
                                    (self.x_grid, self.y_grid),method='cubic')
        else:
            zave = point_heights.mean()
            heightgrid=np.zeros_like(self.x_grid);
            heightgrid[:,:]=zave
        return heightgrid

    def SaveGeoTiff(self, result2D, filename):
        assert((result2D.shape[0]==self.npix_y) & (result2D.shape[1]==self.npix_x))
        xres = (self.xmax - self.xmin) / float(self.npix_x)
        yres = (self.ymax - self.ymin) / float(self.npix_y)
        geotransform = (self.xmin, xres, 0, self.ymax, 0, -yres)
        from osgeo import gdal
        from osgeo import osr
        dst_ds = gdal.GetDriverByName('GTiff').Create(filename, self.npix_x, self.npix_y, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).WriteArray(result2D)
        dst_ds.FlushCache()
        dst_ds = None
    

class HeightMap:
    def __init__(self, webmerc: WebMercatorImage):
        self.webmerc = webmerc
        self.heightmap = None
    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def load(cls, filename, cloudImage = None):
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        return d


class Camera:
    def __init__(self, cloudImage):
        self.cloudImage = cloudImage
        self.camera_enu = None
        self.camera_ecef = None
    def __getstate__(self):
        state = {}
        for ckey, cam in [('ENU',self.camera_enu), ('ECEF',self.camera_ecef)]:
            if cam is not None:
                keys = cam.parameters.parameters.keys()
                state[ckey] = {key: getattr(cam, key) for key in keys}
        return state
    def __setstate__(self, state):
        if 'camera_enu' in state:
            # fallback to default picklel saved with previous versions
            self.__dict__ = state
        else:
            self.camera_enu = None
            self.camera_ecef = None
            for ckey in ['ENU','ECEF']:
                if ckey in state:
                    cam = ct.Camera(ct.RectilinearProjection(),
                          ct.SpatialOrientation(),
                          ct.BrownLensDistortion())
                    for key in state[ckey]:
                        setattr(cam, key, state[ckey][key])
                    if ckey=='ENU':
                        self.camera_enu=cam
                    else:
                        self.camera_ecef=cam

    def residual(self, x, camera, space_coords, pxls):
        camera.heading_deg=x[0]
        camera.tilt_deg=x[1]
        camera.roll_deg=x[2]
        camera.focallength_x_px=x[3]
        camera.focallength_y_px=x[4]
        camera.center_x_px=x[5]
        camera.center_y_px=x[6]
        camera.k1=x[7]
        return np.sqrt(np.mean((camera.imageFromSpace(space_coords)-pxls)**2))

    def _fit_metropolis(self):
        camera1 = self.camera_enu
        trace = camera1.metropolis([
                ct.FitParameter("tilt_deg", lower=0, upper=180, value=90),
                ct.FitParameter("heading_deg", lower=-180, upper=180, value=0),
                ct.FitParameter("roll_deg", lower=-180, upper=180, value=0),
                ], iterations=1e4)
        trace = camera1.metropolis([
                ct.FitParameter("focallength_x_px", lower=2000, upper=10000, value=4000),
                ct.FitParameter("focallength_y_px", lower=2000, upper=10000, value=4000),
                ], iterations=1e4)
        trace = camera1.metropolis([
                ct.FitParameter("center_x_px", lower=1500, upper=4000, value=3008),
                ct.FitParameter("center_y_px", lower=1000, upper=3000, value=2008)
                ], iterations=1e4)
        trace = camera1.metropolis([
                ct.FitParameter("k1", lower=0, upper=1, value=0),
                ], iterations=1e4)
        trace = camera1.metropolis([
                ct.FitParameter("tilt_deg", lower=0, upper=180, value=80),
                ct.FitParameter("heading_deg", lower=-180, upper=180, value=-77),
                ct.FitParameter("roll_deg", lower=-180, upper=180, value=0),
                ], iterations=3e3)
        trace = camera1.metropolis([
                ct.FitParameter("focallength_x_px", lower=3000, upper=10000, value=4000),
                ct.FitParameter("focallength_y_px", lower=3000, upper=10000, value=4000),
                ], iterations=3e3)
        trace = camera1.metropolis([
                ct.FitParameter("center_x_px",  lower=2000, upper=4000, value=3008),
                ct.FitParameter("center_y_px", lower=1000, upper=3000, value=2008)
                ], iterations=3e3)
        trace = camera1.metropolis([
                ct.FitParameter("k1", lower=0, upper=1, value=0),
                ], iterations=2e3)

    def Fit(self, method='optnew', distortion=False, centers=True, separate_x_y=True):
        #exif = utils.getExifTags(self.cloudImage.filename)
        #focallength = exif.get('FocalLength', 24)
        sensor_size = (36, 24)
        focallength = utils.getExifEquivalentFocalLength35mm(self.cloudImage.filename)

        # if focal length given in exif specify range f/2 - 2*f
        f_bounds = np.array([12, 600]) if focallength is None else np.array([focallength/2.0, focallength*2.0])
        f_bounds = f_bounds/sensor_size[0]*self.cloudImage.imagearray.shape[1]


        focallength = 24 if focallength is None else focallength
        focallength_px=focallength/sensor_size[0]*self.cloudImage.imagearray.shape[1]

        image_size = (self.cloudImage.imagearray.shape[1],  self.cloudImage.imagearray.shape[0])

        cx_bounds = [0, self.cloudImage.imagearray.shape[1]]
        cy_bounds = [0, self.cloudImage.imagearray.shape[0]]

        self.camera_enu = ct.Camera(ct.RectilinearProjection(focallength_mm=focallength,
                                         sensor=sensor_size,
                                         image=image_size),
                  ct.SpatialOrientation(elevation_m=0),
                  ct.BrownLensDistortion())

        enu_unit_coords = self.cloudImage.get_stars_enu_unit_coords()
        pxls = self.cloudImage.getPixelCoords()
        self.camera_enu.addLandmarkInformation(pxls, enu_unit_coords, [0.01, 0.01, 0.01])
        self.camera_enu.pos_x_m=0
        self.camera_enu.pos_y_m=0
        self.camera_enu.elevation_m=0
        if method=='optnew':
            self.camera_enu = optimize_camera.OptimizeCamera(self.camera_enu, enu_unit_coords, pxls, distortion=distortion, centers=centers,separate_x_y=separate_x_y,
                                                             f_bounds=f_bounds, cx_bounds=cx_bounds, cy_bounds=cy_bounds)
        elif method=='opt':
            optres = scipy.optimize.minimize(optimize_camera.ResRot, [1,0,0,focallength_px,focallength_px,
                                                                      (cx_bounds[0]+cx_bounds[1])/2.0,
                                                                      (cy_bounds[0]+cy_bounds[1])/2.0,0], args=(  self.camera_enu, enu_unit_coords, pxls), method='SLSQP',
                                bounds=[[-7,7],[-7,7],[-7,7], f_bounds, f_bounds, cx_bounds,cy_bounds,[0,1]])
            print(optres)
        else:
            self._fit_metropolis()

        calculated_star_px_coords = self.camera_enu.imageFromSpace(enu_unit_coords)
        residuals = calculated_star_px_coords - pxls
        print('ENU camera res:',np.sqrt(np.mean(residuals**2)))
        for sr, delta in zip(self.cloudImage.starReferences, residuals):
            print(sr, np.abs(delta))

        self.camera_ecef =  self.camera_ecef_from_camera_enu(self.camera_enu)

        ecef_unit = pymap3d.enu2ecef(enu_unit_coords[:,0], enu_unit_coords[:,1], enu_unit_coords[:,2],
                                      self.cloudImage.location.lat.value,self.cloudImage.location.lon.value,self.cloudImage.location.height.value)
        ecef_unit =np.array(ecef_unit).T

        print('ECEF camera res:',np.sqrt(np.mean((self.camera_ecef.imageFromSpace(ecef_unit)-pxls)**2)))


        return

    def camera_ecef_from_camera_enu(self, camera_enu):
        import copy
        # ja lieto astropy objektus, tad pārveidot no radiāniem uz grādiem nevajag
        sinlon, coslon = np.sin(self.cloudImage.location.lon).value,np.cos(self.cloudImage.location.lon).value
        sinlat, coslat = np.sin(self.cloudImage.location.lat).value,np.cos(self.cloudImage.location.lat).value
        rotMatr_uvw_enu=np.array([
            [-sinlon, coslon, 0],
            [-sinlat*coslon, -sinlat*sinlon, coslat],
            [coslat*coslon, coslat*sinlon, sinlat]
        ])
        angl = optimize_camera.Orientation_fromRotation(optimize_camera.Rotation_fromOrientation(camera_enu)*Rotation.from_matrix(rotMatr_uvw_enu))

        camera_ecef = ct.Camera(copy.deepcopy(camera_enu.projection),
                                ct.SpatialOrientation(**angl),
                                copy.deepcopy(camera_enu.lens))
        camera_ecef.pos_x_m=self.cloudImage.location.x.value
        camera_ecef.pos_y_m=self.cloudImage.location.y.value
        camera_ecef.elevation_m=self.cloudImage.location.z.value
        return camera_ecef

    def save(self, filename):
        self.camera_enu.save(os.path.splitext(filename)[0]+'_enu.json')
        self.camera_ecef.save(os.path.splitext(filename)[0]+'_ecef.json')
    def load(self, filename):
        if os.path.exists(os.path.splitext(filename)[0]+'_enu.json'):
            self.camera_enu = optimize_camera.load_camera(os.path.splitext(filename)[0]+'_enu.json')
        if os.path.exists(os.path.splitext(filename)[0]+'_ecef.json'):
            self.camera_ecef = optimize_camera.load_camera(os.path.splitext(filename)[0]+'_ecef.json')

    def imageFromECEF(self, xyz):
        return self.camera_ecef.imageFromSpace(xyz)
    def imageFromAltAz(self, az,alt):
        enu = pymap3d.aer2enu(az,alt,1.0)
        return self.camera_enu.imageFromSpace(enu)

    def get_azimuth_elevation_rotation(self):
        camera_enu=self.camera_enu
        if camera_enu.tilt_deg<0:
            azimuth =180+camera_enu.heading_deg if camera_enu.heading_deg<0 else camera_enu.heading_deg-180
            tilt =-camera_enu.tilt_deg
            roll =camera_enu.roll_deg-180 if camera_enu.roll_deg>0 else 180+camera_enu.roll_deg
        else:
            azimuth =camera_enu.heading_deg
            tilt =camera_enu.tilt_deg
            roll =camera_enu.roll_deg
        # tilt - 0 deg down, zenith distance=180-tilt, elevation=tilt-90
        return azimuth, tilt-90, roll
    def get_focal_lengths_mm(self):
        camera_enu=self.camera_enu
        
        return camera_enu.focallength_x_px * 36.0 / camera_enu.image_width_px, camera_enu.focallength_y_px * 36.0 / camera_enu.image_width_px
    @classmethod
    def ecef_from_enu(cls, camera_enu, lat_deg, lon_deg, height_m):
        import copy
        # ja lieto astropy objektus, tad pārveidot no radiāniem uz grādiem nevajag
        lon, lat = np.radians(lon_deg), np.radians(lat_deg)
        sinlon, coslon = np.sin(lon),np.cos(lon)
        sinlat, coslat = np.sin(lat),np.cos(lat)
        rotMatr_uvw_enu=np.array([
            [-sinlon, coslon, 0],
            [-sinlat*coslon, -sinlat*sinlon, coslat],
            [coslat*coslon, coslat*sinlon, sinlat]
        ])
        angl = optimize_camera.Orientation_fromRotation(optimize_camera.Rotation_fromOrientation(camera_enu)*Rotation.from_matrix(rotMatr_uvw_enu))

        camera_ecef = ct.Camera(copy.deepcopy(camera_enu.projection),
                                ct.SpatialOrientation(**angl),
                                copy.deepcopy(camera_enu.lens))
        x, y, z = pymap3d.geodetic2ecef(lat_deg, lon_deg, height_m)
        camera_ecef.pos_x_m=x
        camera_ecef.pos_y_m=y
        camera_ecef.elevation_m=z
        return camera_ecef
    @classmethod
    def make_cameras(cls, lat_deg, lon_deg, height_m, width_px, height_px, focallength_35mm, azimuth, elevation, rotation):
        camera_enu = ct.Camera(ct.RectilinearProjection(focallength_mm=focallength_35mm,
                                             sensor=(36,24),
                                             image=(width_px, height_px)),
                               ct.SpatialOrientation(
                                   heading_deg = azimuth,
                                   tilt_deg = 90 + elevation,
                                   roll_deg = rotation,
                                   elevation_m=0.0
                               ),  ct.BrownLensDistortion())
        camera_enu.pos_x_m=0
        camera_enu.pos_y_m=0
        camera_enu.elevation_m=0

        camera_ecef=Camera.ecef_from_enu(camera_enu, lat_deg, lon_deg, height_m)
        return camera_enu, camera_ecef
    @classmethod
    def GetAltAzGrid_fromcamera(cls, width_px, height_px, camera_enu):
        i_grid, j_grid = np.meshgrid(np.arange(width_px),np.arange(height_px))
        grid_points=np.array([i_grid.flatten(), j_grid.flatten()]).T
        enu = camera_enu.spaceFromImage(grid_points, D=1)
        azalt = pymap3d.enu2aer(*enu.T)[0:2]
        azalt = np.reshape(np.array(azalt),(2,i_grid.shape[0],i_grid.shape[1]))
        az_min, az_max=azalt[0].min(), azalt[0].max()
        if az_max-az_min>180:
            azalt[0]=np.where(azalt[0]>180,azalt[0]-360,azalt[0])
        return azalt


class CloudImage:
    timezone = pytz.timezone('Europe/Riga')
    @classmethod
    def load(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)

        #d.LoadCamera(filename)
        return d

    def __init__(self, code, filename):
        self.code = code
        self.starReferences :list[StarReference] = []
        self.filename = os.path.abspath(filename)
        self.location: astropy.coordinates.EarthLocation = astropy.coordinates.EarthLocation(0,0)
        self.date: astropy.time.Time = astropy.time.Time(datetime.datetime.now())
        self.__imagearray = None
    def __str__(self):
        return self.__dict__.__str__()
    def __repr__(self):
        return self.__dict__.__repr__()

    @property
    def imagearray(self):
        self.LoadImage()
        return self.xxx

    def LoadImage(self, reload = False):
        if not hasattr(self, 'xxx') or self.xxx is None or reload:
            print('Loading image:', os.path.split(self.filename)[1])
            self.xxx = skimage.io.imread(self.filename)
    def imageArrayGrid(self):
        return np.meshgrid(np.arange(self.imagearray.shape[1]),np.arange(self.imagearray.shape[0]))

    @classmethod
    def initialize(cls, fn_registrs, kods):
        saraksts = pd.read_excel(fn_registrs, sheet_name = 'Registrs', index_col=0)
        zvaigznes = pd.read_excel(fn_registrs, sheet_name = 'Zvaigznes', index_col=0)
        fn_image = f"{saraksts.loc[kods]['Katalogs']}\\{saraksts.loc[kods]['Fails']}"
        c = CloudImage(kods, fn_image)
        c.setDateFromExif()
        lat, lon = np.array(saraksts.loc[kods]['Koordinātes'].split(',')).astype('float')
        c.setLocation(lat=lat, lon=lon)
        x = zvaigznes.loc[kods].dropna()
        stars = x.index
        pixels = np.array(x.str.split(',').to_list()).astype('float')
        c.setStarReferences(stars, pixels)
        return c
    @classmethod
    def from_files(cls, case_id, filename_jpg, filename_stars, lat, lon):
        cldim = CloudImage(case_id, filename_jpg)
        cldim.setDateFromExif()
        if lat is not None and lon is not None:
            cldim.setLocation(lat=lat, lon=lon)
        else:
            cldim.setLocationExif()
        print('UTC:', cldim.date)
        print(cldim.location.to_geodetic())
        # uzstādām zvaigžņu sarakstu
        try:
            df = pd.read_csv(filename_stars, sep='\t', header=None)
            # zvaigžņu nosaukumi pirmajā kolonā
            starnames = df[0]
            # atbilstošās pikseļu koordinātes otrajā un trešajā kolonā
            pixels=np.array(df[[1,2]])
        except:
            starnames=[]
            pixels=[]
            
        cldim.setStarReferences(starnames, pixels)
        # izdrukājam zvaigžņu ekvatoriālās un pikseļu koordinātes pārbaudes nolūkos
        print(cldim.getSkyCoords())
        print(cldim.getPixelCoords())               
        return cldim    
    def __getstate__(self):
        state = self.__dict__
        invalid = {"xxx", "_aazgrid", "_radecgrid","_AAZImage"}
        return {x: state[x] for x in state if x not in invalid}

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def setDate(self, datetime: datetime.datetime, tz=timezone):
        date_localized=tz.localize(datetime)
        # iegūstam astropy laika objektu, pievienojam images sarakstam
        self.date = astropy.time.Time(date_localized)
        self.prepareCoordinateSystems()
    def setDateFromExif(self):
        try:
            d = utils.dateFromExif(self.filename)
            self.setDate(d)
        except:
            print('WARNING: date does not exist in EXIF')
            pass
    def setLocation(self, lon: float, lat: float, height: float = 0):
        # astropy EarthLocation objekts, kuru inicializējam no novērotāja koordinātēm
        self.location = astropy.coordinates.EarthLocation(lon = lon, lat = lat, height = height)
        self.prepareCoordinateSystems()
    def setLocationExif(self):
        lat, lon = utils.getExifLatLon(self.filename)
        print('EXIF latlon', lat, lon)
        if lon is not None and lat is not None:
            self.setLocation(lon, lat)
        else:
            raise Exception("Image contains no EXIF GPS data, latitude and longitude must be explicitly specified")
    def prepareCoordinateSystems(self):
        # horizontālā koordinātu sistēma, atbilstoši novērotāja pozīcijai un laikam (tie iegūti iepriekš)
        self.altaz =astropy.coordinates.AltAz(obstime=self.date, location=self.location)
        # ITRS koordinātu sistēma, kas atbilst novērošanas laikam (būs nepieciešama tālāk, ģoecentrikso koordināšu noteikšanai)
        self.itrs = astropy.coordinates.ITRS(obstime=self.date)
    def setStarReferences(self, starList, pixelList):
        assert(len(starList)==len(pixelList))
        self.starReferences = [StarReference(star, pixel) for star, pixel in zip(starList, pixelList)]
        for r in self.starReferences:
            r.getSkyCoord()
    def saveStarReferences(self, filename):
        
        df=pd.DataFrame({'name':[r.name for r in self.starReferences],
                         'ix':[r.pixelcoords[0] for r in self.starReferences],
                         'iy':[r.pixelcoords[1] for r in self.starReferences]})
        df.to_csv(filename, sep='\t', header=None, index=False)
        
    def getPixelCoords(self):
        return np.array([r.pixelcoords for r in self.starReferences])
    def getSkyCoords(self):
        try:
            return astropy.coordinates.SkyCoord([r.skycoord for r in self.starReferences])
        except:
            return []
    def GetWCS(self, sip_degree = 2,  fit_parameters={}):
        # izveidojam WCS priekš attēla (world coordinate system)
        # sip ir 'Simple Imaging Polynomial' https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/shupeADASS.pdf
        # tas ļauj transformēt attēlu un ņem vērā dažādus potenciālos attēlu kropļojumus
        # šeit uzdodam polinoma kārtu, atkarībā no zvaigžņu izvietojuma kvalitātes uz katra no attēliem, šo vajadzētu mainīt
        # tāpēc arī ieviesu ['wcs']['sip_degree'] mainīgo pie katra no attēliem.
        # (Piem. nevar likt augstu kārtu polionomam, ja references zvaigznes ir novietotas gandrīz uz vienas līnijas)

        # wcs objekta atrašana no zvaigznēm atbilsotšajām pikseļu koordinātēm
        pixelcoords = self.getPixelCoords()
        skycoords = self.getSkyCoords()
        wcs = astropy.wcs.utils.fit_wcs_from_points((pixelcoords[:,0], pixelcoords[:,1]),
                astropy.coordinates.SkyCoord(skycoords), sip_degree=sip_degree,**fit_parameters)
        print(wcs)
        self.wcs=wcs

    def TestStarFit(self):
        # test fit
        wcs = self.wcs
        for starref in self.starReferences:
            px = starref.pixelcoords
            sc=wcs.pixel_to_world(*px)
            x, y =wcs.world_to_pixel(starref.skycoord)
            dist=np.sqrt((px[0]-x)**2+(px[1]-y)**2)
            print(starref.name, sc.separation(starref.skycoord), dist)


    @property
    def radecgrid(self):
        self.GetImageRaDecGrid()
        return self._radecgrid

    def GetImageRaDecGrid(self, reload=False):
        if not hasattr(self, '_radecgrid') or self._radecgrid is None or reload:
            print('Calculate RaDec grid')
            self._radecgrid = calculations.GetImageRaDecGrid(self.imagearray, self.wcs)
    @property
    def aazgrid(self):
        self.GetAltAzGrid()
        return self._aazgrid
    def GetAltAzGrid(self, reload=False):
        if not hasattr(self, '_aazgrid') or self._aazgrid is None or reload:
            print('Calculate AltAz grid')
            self._aazgrid = self.radecgrid.transform_to(self.altaz)
    def GetAltAzGrid_fromcamera(self):
        return Camera.GetAltAzGrid_fromcamera(self.imagearray.shape[1], self.imagearray.shape[0], self.camera.camera_enu)

    @property
    def AAZImage(self):
        self.Initialize_AltAzImage()
        return self._AAZImage

    def Initialize_AltAzImage(self, width=None,  height=None, reload=False):
        if not hasattr(self, '_AAZImage') or self._AAZImage is None or reload:
            self._AAZImage = AAZImage(self)
            self._AAZImage.Initialize(width, height)

    def PrepareAltAZImage_throughRA(self):
        self.AAZImage.Prepare_throughRA(self)

    def PrepareCamera(self, method='optnew', **params):
        self.camera = Camera(self)
        self.camera.Fit(method, **params)
    def LoadCamera(self, filename):
        self.camera = Camera(self)
        self.camera.load(filename)
    def SaveCamera(self, filename):
        self.camera.save(filename)

# funckija atrod pozīcijas ģeocentriskās koordinātes punktam, kas atrodas attālumā d1 (kilometros) no novērotāja un virzienā, ko
# definē aazc (augstumu un azimutu  sistēma ar novērotāju centrā)
    @classmethod
    def GetEarthLocation(cls, aazc, itrs, d1):
        # izveidojam punktu d1 kilometru attālumā no aazc.location virzienā uz az, alt
        coord = astropy.coordinates.AltAz(location = aazc.location, obstime = aazc.obstime, az=aazc.az, alt=aazc.alt,
                                          distance = d1 * u.km)
        # iegūstam ģeocentriskās koordinātes šim punktam
        ploc1=coord.transform_to(itrs).earth_location
        return ploc1
    # funckija nosaka attēlam atbilstošās horizontālās koordinātes (AltAz), punktā ar pikseļu koordinātēm
    def GetAAzCoord(self, pixelcoords):
        wcs=self.wcs
        skycoord = wcs.pixel_to_world(*pixelcoords)
        aazc = skycoord.transform_to(self.altaz)
        return aazc
    # sākotnēja metode
    @classmethod
    def GetDirection(cls, p, itrs):
        d=0.001
        #print('P1' ,p1.location)
        p2=CloudImage.GetEarthLocation(p, itrs, d)
        #print('PS:',(p2.x-p.location.x).value)
        direction = np.array([(p2.x-p.location.x).value,(p2.y-p.location.y).value,(p2.z-p.location.z).value])/(d)
        return direction

    def get_stars_enu_unit_coords(self):
        stars_altaz = self.getSkyCoords().transform_to(self.altaz)
        enu_unit_coords = pymap3d.aer2enu(stars_altaz.az.value, stars_altaz.alt.value,1)
        enu_unit_coords=np.array(enu_unit_coords).T
        return enu_unit_coords

class Reprojector_to_Camera:
    def __init__(self, cldim: CloudImage, camera_ecef):
        self.cloudImage = cldim
        self.camera = camera_ecef
    def prepare_reproject(self, height_km):
        width2, height2 = self.camera.image_width_px, self.camera.image_height_px
        i_grid, j_grid = np.meshgrid(np.arange(width2),np.arange(height2))
        grid_points=np.array([i_grid.flatten(), j_grid.flatten()]).T
        cam2 = self.camera
        center, rays = cam2.getRay(grid_points, normed=True)
        ray_coords=rays.T
        xyz = geoutils.los_to_constant_height_surface(*center,*ray_coords, height_km*1000)
        xyz=xyz.T
        cam1 = self.cloudImage.camera.camera_ecef
        image_pxls = cam1.imageFromSpace(xyz)
        image_pxls = np.reshape(image_pxls, (i_grid.shape[0],i_grid.shape[1],2))
        i_pxls, j_pxls = image_pxls[:,:,0],image_pxls[:,:,1]
        width1, height1 = self.cloudImage.imagearray.shape[1], self.cloudImage.imagearray.shape[0]
        maskpix = ~np.isnan(i_pxls) & ~np.isnan(j_pxls)
        i_pxls=np.round(i_pxls).astype('int')
        j_pxls=np.round(j_pxls).astype('int')
        maskpix = maskpix &  (i_pxls>=0) &  (i_pxls<width1)  & (j_pxls>=0) & (j_pxls<height1)
        self.maskpix = maskpix
        self.i_pxls = i_pxls
        self.j_pxls = j_pxls
        self.i_grid = i_grid
        self.j_grid = j_grid
    def Fill_projectedImage(self):
        width2, height2 = self.camera.image_width_px, self.camera.image_height_px
        projected_image=np.zeros(shape=(height2, width2, 3), dtype='uint8')
        projected_image[self.j_grid[self.maskpix], self.i_grid[self.maskpix]]=self.cloudImage.imagearray[self.j_pxls[self.maskpix], self.i_pxls[self.maskpix]]
        return projected_image
    def Fill_projectedImageMasked(self,):
        img = self.Fill_projectedImage()
        alpha = ((1-((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0)))*255).astype('uint8')
        masked_img = np.append(img, alpha[:,:,np.newaxis], axis=2)
        return masked_img


class Reprojector_12:
    def __init__(self, cloudImagePair):
        self.cloudImagePair = cloudImagePair
    def prepare_reproject_1_2(self, height_km):
        cp = self.cloudImagePair
        i_grid, j_grid = cp.cloudImage2.imageArrayGrid()
        grid_points=np.array([i_grid.flatten(), j_grid.flatten()]).T
        cam2 = cp.cloudImage2.camera.camera_ecef
        center, rays = cam2.getRay(grid_points, normed=True)
        ray_coords=rays.T
        xyz = geoutils.los_to_constant_height_surface(*center,*ray_coords, height_km*1000)
        xyz=xyz.T
        cam1 = cp.cloudImage1.camera.camera_ecef
        image_pxls = cam1.imageFromSpace(xyz)
        image_pxls = np.reshape(image_pxls, (i_grid.shape[0],i_grid.shape[1],2))
        i_pxls, j_pxls = image_pxls[:,:,0],image_pxls[:,:,1]
        width, height = cp.cloudImage1.imagearray.shape[1], cp.cloudImage1.imagearray.shape[0]
        maskpix = ~np.isnan(i_pxls) & ~np.isnan(j_pxls)
        i_pxls=np.round(i_pxls).astype('int')
        j_pxls=np.round(j_pxls).astype('int')
        maskpix = maskpix &  (i_pxls>=0) &  (i_pxls<width)  & (j_pxls>=0) & (j_pxls<height)
        self.maskpix = maskpix
        self.i_pxls = i_pxls
        self.j_pxls = j_pxls
        self.i_grid = i_grid
        self.j_grid = j_grid
    def Fill_projectedImage(self):
        cp = self.cloudImagePair
        width, height = cp.cloudImage2.imagearray.shape[1], cp.cloudImage2.imagearray.shape[0]
        projected_image=np.zeros(shape=(height, width, 3), dtype='uint8')
        projected_image[self.j_grid[self.maskpix], self.i_grid[self.maskpix]]=cp.cloudImage1.imagearray[self.j_pxls[self.maskpix], self.i_pxls[self.maskpix]]
        return projected_image
    def Fill_projectedImageMasked(self,):
        img = self.Fill_projectedImage()
        alpha = ((1-((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0)))*255).astype('uint8')
        masked_img = np.append(img, alpha[:,:,np.newaxis], axis=2)
        return masked_img


class CloudImagePair:
    def __init__(self, cloudImage1, cloudImage2):
        self.cloudImage1 = cloudImage1
        self.cloudImage2 = cloudImage2
        self.reproject = Reprojector_12(self)
        self.correspondances = [np.empty(shape=(0,2)),np.empty(shape=(0,2))]

    def InitCameraGroup(self):
        camera1 = self.cloudImage1.camera.camera_ecef
        camera2 = self.cloudImage2.camera.camera_ecef
        self.cameraGroup  = ct.CameraGroup([camera1.projection, camera2.projection],
                                           [camera1.orientation, camera2.orientation],
                                           [camera1.lens, camera2.lens])
    def GetEpilinesAtHeightIntervalOnGrid(self, z_arr_km, grid_px_distance=200, pointsInCamera1 = True):
        if pointsInCamera1:
            ii, jj = np.meshgrid(np.arange(0,self.cloudImage1.imagearray.shape[1],grid_px_distance),np.arange(0,self.cloudImage1.imagearray.shape[0],grid_px_distance))
        else:
            ii, jj = np.meshgrid(np.arange(0,self.cloudImage2.imagearray.shape[1],grid_px_distance),np.arange(0,self.cloudImage2.imagearray.shape[0],grid_px_distance))
        pp = np.array([ii.flatten(), jj.flatten()])
        point_in1=pp.T
        return  point_in1, self.GetEpilinesAtHeightInterval(z_arr_km, point_in1, pointsInCamera1)
    def GetEpilinesAtHeightInterval(self, z_arr_km, point_array, pointsInCamera1 = True):
        point_in1 = point_array
        if pointsInCamera1:
            camera1 = self.cloudImage1.camera.camera_ecef
            camera2 = self.cloudImage2.camera.camera_ecef
        else:
            camera2 = self.cloudImage1.camera.camera_ecef
            camera1 = self.cloudImage2.camera.camera_ecef
        center, rays = camera1.getRay(point_in1, normed=True)
        ray_coords=rays.T
        image_pxls=[]
        for z in z_arr_km:
            xyz1 = geoutils.los_to_constant_height_surface(*center,*ray_coords, z*1000)
            xyz1=xyz1.T
            image_pxls1 = camera2.imageFromSpace(xyz1)
            image_pxls.append(image_pxls1)
        epilines = np.swapaxes(np.array(image_pxls),0,1)
        return  epilines

    def GetHeightPoints(self, corresponding1, corresponding2, max_intrinsic_error=0.5, max_epiline_distance_error=1.0):
        '''
            max_intrinsic_error (in km height/px)
            max_epiline_distance_error (in km height)
        '''
        self.InitCameraGroup()
        z1=75
        z2=90
        cgr = self.cameraGroup
        xyz=cgr.spaceFromImages(corresponding1, corresponding2)
        rayminimaldistance=cgr.discanteBetweenRays(corresponding1, corresponding2)
        llh = pymap3d.ecef2geodetic(xyz[:,0],xyz[:,1],xyz[:,2])
        epilines = self.GetEpilinesAtHeightInterval([z1,z2],corresponding1)
        # 1px /(length of epilines in px)/(z2_km-z1_km)
        z_intrinsic_error = (z2-z1)/np.linalg.norm(epilines[:,1,:]-epilines[:,0,:], axis=1)
        valid = (z_intrinsic_error<max_intrinsic_error) & (rayminimaldistance/1000<max_epiline_distance_error)
        return llh, rayminimaldistance, z_intrinsic_error, valid

    def GetEpilineLengthPerHeight(self):
        r1 = Reprojector_12(self)
        r2 = Reprojector_12(self)
        z1=75
        z2=90
        r1.prepare_reproject_1_2(z1)
        r2.prepare_reproject_1_2(z2)
        r1i=np.ma.masked_array(r1.i_pxls, mask  =~r1.maskpix)
        r1j=np.ma.masked_array(r1.j_pxls, mask  =~r1.maskpix)
        r2i=np.ma.masked_array(r2.i_pxls, mask  =~r2.maskpix)
        r2j=np.ma.masked_array(r2.j_pxls, mask  =~r2.maskpix)
        px_per_km = np.sqrt((r1i-r2i)**2+(r1j-r2j)**2)/(z2-z1)
        return px_per_km

    def LoadCorrespondances(self, filename):
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep='\t', header=[0,1])
            pts1=np.array(df.iloc[:,[0,1]])
            pts2=np.array(df.iloc[:,[2,3]])
            self.correspondances = [pts1, pts2]
        else:
            print(f'ATBILSTības fails {filename} neeksistē!')
            self.correspondances = [np.empty(shape=(0,2)),np.empty(shape=(0,2))]
    def SaveCorrespondances(self, filename):
        ll=min(len(self.correspondances[0]),len(self.correspondances[1]))
        self.correspondances[0]=self.correspondances[0][0:ll]
        self.correspondances[1]=self.correspondances[1][0:ll]
        pd.DataFrame({(self.cloudImage1.code,'i'): self.correspondances[0][:,0],
                      (self.cloudImage1.code,'j'): self.correspondances[0][:,1],
                      (self.cloudImage2.code,'i'): self.correspondances[1][:,0],
                      (self.cloudImage2.code,'j'): self.correspondances[1][:,1],
                      }).to_csv(filename, sep='\t', index=False)
    # sākotnējā metode
    def GetReferencePointPositions(self):
        result = []
        for i in range(len(self.correspondances[0])):
            lll = []
            aazc = self.cloudImage1.GetAAzCoord(self.correspondances[0][i])
            lll.append((aazc, self.cloudImage1.itrs))
            aazc = self.cloudImage2.GetAAzCoord(self.correspondances[1][i])
            lll.append((aazc, self.cloudImage2.itrs))
            (dist1, dist2), minerror, (p1, p2)  = CloudImagePair.GetRayConvergence(lll[0][0], lll[1][0], lll[0][1], lll[1][1])
            #print(dist1, dist2, minerror)

            #loc1 = CloudImage.GetEarthLocation(lll[0][0],  lll[0][1], dist1).to_geodetic()
            #loc2 = CloudImage.GetEarthLocation(lll[1][0], lll[1][1], dist2).to_geodetic()
            loc1, loc2 = pymap3d.ecef2geodetic(*p1),pymap3d.ecef2geodetic(*p2)
            loccenter = pymap3d.ecef2geodetic(*((p1+p2)*0.5))
            print(loc1,loc2, loccenter)
            result.append(((loc1, loc2), (dist1,dist2), minerror, loccenter))
        result = pd.DataFrame([{'nr': i, 'lat1':r[0][0][0], 'lon1':r[0][0][1],'height1':r[0][0][2],
                              'lat2':r[0][1][0], 'lon2':r[0][1][1],'height2':r[0][1][2],
                              'dist1':r[1][0], 'dist2':r[1][1],
                              'raydistance': r[2],
                              'lat':r[3][0], 'lon':r[3][1],'height':r[3][2],
                             }  for r in result])
        return result

    # oriģināla metode neizmatojot kameras objektu
    @classmethod
    def GetRayConvergence(cls, pos1, pos2, itrs1, itrs2, method=None, verbose=False):
        p01=np.array([pos1.location.x.value,pos1.location.y.value,pos1.location.z.value])
        p02=np.array([pos2.location.x.value,pos2.location.y.value,pos2.location.z.value])
        direction1 = CloudImage.GetDirection(pos1,itrs1)
        direction2 = CloudImage.GetDirection(pos2,itrs2)
        matr=[[np.dot(direction1, direction1),-np.dot(direction1, direction2)],
          [np.dot(direction2, direction1),-np.dot(direction2, direction2)]]
        minv=np.linalg.inv(matr)
        (dist1, dist2)=np.dot(minv,np.array([np.dot(p02-p01,direction1),np.dot(p02-p01,direction2)]))
        print(dist1,dist2)
        p1 = p01+dist1*direction1
        p2 = p02+dist2*direction2
        delta = np.linalg.norm(p1-p2)
        return (dist1/1000, dist2/1000), delta/1000, (p1,p2)

