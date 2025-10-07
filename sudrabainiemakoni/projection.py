from abc import ABC, abstractmethod
import camera
import pymap3d
import numpy as np
from sudrabainiemakoni import geoutils
class ProjectionImage(ABC):
    def __init__(self, cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km):
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax
        self.pixel_per_km = pixel_per_km
        bounds = [(lonmin,latmin),(lonmin,latmax),(lonmax,latmin),(lonmax,latmax)]
        x, y = self.latlon_to_xy([b[1] for b in bounds],[b[0] for b in bounds])
        self.xmin = min(x)
        self.xmax = max(x)
        self.ymin = min(y)
        self.ymax = max(y)
        self.npix_x, self.npix_y = int((self.xmax-self.xmin)/(pixel_per_km*1000)),int((self.ymax-self.ymin)/(pixel_per_km*1000))
        self.cloudImage = cloudImage
        self.initialize()
        
    @abstractmethod
    def xy_to_latlon(self, x, y):
        lat = y
        lon = x
        return lat, lon
        
    @abstractmethod
    def latlon_to_xy(self, lat, lon):
        x = lon
        y = lat
        return x, y
    
    def initialize(self):
        # coordinate arrays in projection coordinates
        self.x_arr = np.linspace(self.xmin,self.xmax, self.npix_x)
        self.y_arr = np.linspace(self.ymax,self.ymin, self.npix_y)
        # projection coordinates grid
        self.x_grid, self.y_grid = np.meshgrid(self.x_arr, self.y_arr)
        # ij grid 
        self.ix_grid, self.iy_grid = np.meshgrid(np.arange(0, self.npix_x), np.arange(0, self.npix_y))
        # latitude and longitude grid
        self.lat_grid, self.lon_grid = self.xy_to_latlon(self.x_grid.flatten(),self.y_grid.flatten())
        self.lon_grid = self.lon_grid.reshape(self.x_grid.shape)
        self.lat_grid = self.lat_grid.reshape(self.x_grid.shape)
        
    
    def prepare_reproject_from_camera(self, height_km, cam: camera.Camera = None):
        if cam is None:
            cam = camera.Cameratransform(self.cloudImage.camera.camera_ecef)
        height = height_km * 1000
        x,y,z = pymap3d.geodetic2ecef(self.lat_grid, self.lon_grid, height)
        xyz = np.array([x.flatten(),y.flatten(),z.flatten()]).T
        image_pxls = cam.imageFromSpace(xyz, hide_backpoints=False)
        image_pxls = np.reshape(image_pxls, (x.shape[0],x.shape[1],2))
        self.i_pxls, self.j_pxls = image_pxls[:,:,0],image_pxls[:,:,1]
        width, height = self.cloudImage.imagearray.shape[1], self.cloudImage.imagearray.shape[0]
        self.maskpix = ~np.isnan(self.i_pxls) & ~np.isnan(self.j_pxls)
        self.i_pxls=np.round(self.i_pxls).astype('int')
        self.j_pxls=np.round(self.j_pxls).astype('int')
        self.maskpix = self.maskpix &  (self.i_pxls>=0) &  (self.i_pxls<width)  & (self.j_pxls>=0) & (self.j_pxls<height)
        
    def Fill_projectedImage(self, imagearray=None):
        if imagearray is None:
            imagearray=self.cloudImage.imagearray
        projected_image=np.zeros(shape=(self.npix_y, self.npix_x, 3), dtype='uint8')
        projected_image[self.iy_grid[self.maskpix], self.ix_grid[self.maskpix]]=imagearray[self.j_pxls[self.maskpix], self.i_pxls[self.maskpix]]
        return projected_image
    
    def Fill_projectedImageMasked(self, imagearray=None):     
        img = self.Fill_projectedImage(imagearray)
        alpha = ((1-((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0)))*255).astype('uint8')
        masked_img = np.append(img, alpha[:,:,np.newaxis], axis=2)
        return masked_img
        
    def GetPixelCoords(self, x, y):
        x_pix = (x-self.xmin)/(self.xmax-self.xmin)*self.npix_x
        y_pix = (self.ymax-y)/(self.ymax-self.ymin)*self.npix_y
        return x_pix, y_pix
    
    def GetPixelCoords_LatLon(self, lat, lon):
        x, y = self.latlon_to_xy(lat, lon)
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
        import imageio.v3 as iio
        img = self.Fill_projectedImage()
        iio.imwrite(filename, img)

    def SaveImageTif(self, filename):
        import imageio.v3 as iio
        img = self.Fill_projectedImageMasked()
        iio.imwrite(filename, img)
    
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

    def PrepareHeightMap(self, point_longitudes, point_latitudes, point_heights):
        import pykrige
        assert(len(point_longitudes)==len(point_longitudes))
        assert(len(point_longitudes)==len(point_heights))
        if len(point_longitudes)>=3:
            x, y = self.latlon_to_xy(point_latitudes, point_longitudes)
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

    # Serialization support
    fields = {"lonmin","lonmax","latmin","latmax","pixel_per_km","xmin", "xmax", "ymin", "ymax", "npix_x", "npix_y"}
    
    def __getstate__(self):
        state = self.__dict__
        return {x: state[x] for x in self.fields}
    
    def __setstate__(self, state):
        for x in self.fields:
            setattr(self, x, state[x])
        self.initialize()

    def __str__(self):
        return {k:v for k, v in self.__dict__.items() if k in self.fields}.__str__()
    
    def __repr__(self):
        return {k:v for k, v in self.__dict__.items() if k in self.fields}.__repr__()

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
import pyproj
class ProjectionImagePyproj(ProjectionImage):
    Transformer_LATLON_XY = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")
    Transformer_XY_LATLON = pyproj.Transformer.from_crs("EPSG:3857","EPSG:4326")    
    def __init__(self, proj, cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km):
        self.set_projection(proj)
        super().__init__(cloudImage, lonmin, lonmax, latmin, latmax, pixel_per_km)
    def xy_to_latlon(self, x, y):
        lat, lon = self.Transformer_XY_LATLON.transform(x,y)
        return lat, lon
    
    def latlon_to_xy(self, lat, lon):
        x, y = self.Transformer_LATLON_XY.transform(lat,lon)
        return x, y    
    
    def set_projection(self, proj):
        
        if isinstance(proj, str):
            proj = pyproj.Proj(proj)
        self.proj = proj
        self.Transformer_LATLON_XY = pyproj.Transformer.from_proj("EPSG:4326", proj)
        self.Transformer_XY_LATLON = pyproj.Transformer.from_proj(proj, "EPSG:4326")

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
        # Extract EPSG code from projection
        epsg_code = self.proj.crs.to_epsg()
        if epsg_code:
            srs.ImportFromEPSG(epsg_code)
        else:
            srs.ImportFromProj4(self.proj.crs.to_proj4())
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).WriteArray(result2D)
        dst_ds.FlushCache()
        dst_ds = None

    def SaveGeoTiffRasterio(self, image_data, filename, compress='lzw'):
        """
        Save image data as GeoTIFF using rasterio with proper CRS information.
        
        Args:
            image_data: numpy array with shape (height, width) or (height, width, bands)
            filename: output filename
            compress: compression method ('lzw', 'jpeg', 'deflate', or None)
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
        except ImportError:
            raise ImportError("rasterio is required for SaveGeoTiffRasterio. Install with: pip install rasterio")
        
        # Validate image dimensions
        if len(image_data.shape) == 2:
            height, width = image_data.shape
            count = 1
        elif len(image_data.shape) == 3:
            height, width, count = image_data.shape
        else:
            raise ValueError("Image data must be 2D or 3D numpy array")
        
        assert height == self.npix_y and width == self.npix_x, \
            f"Image dimensions ({height}, {width}) don't match projection dimensions ({self.npix_y}, {self.npix_x})"
        
        # Create transform from bounds
        transform = from_bounds(self.xmin, self.ymin, self.xmax, self.ymax, self.npix_x, self.npix_y)
        
        # Get CRS from projection
        try:
            epsg_code = self.proj.crs.to_epsg()
            if epsg_code:
                crs = CRS.from_epsg(epsg_code)
            else:
                crs = CRS.from_proj4(self.proj.crs.to_proj4())
        except:
            # Fallback to string representation
            crs = CRS.from_string(str(self.proj.crs))
        
        # Set up rasterio profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': image_data.dtype,
            'crs': crs,
            'transform': transform,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
        
        if compress:
            profile['compress'] = compress
        
        # Save the GeoTIFF
        with rasterio.open(filename, 'w', **profile) as dst:
            if count == 1:
                dst.write(image_data, 1)
            else:
                for i in range(count):
                    dst.write(image_data[:, :, i], i + 1)