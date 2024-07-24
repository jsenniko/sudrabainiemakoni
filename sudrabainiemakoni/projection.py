from abc import ABC, abstractmethod
import camera
import pymap3d
import numpy as np
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
        import skimage
        img = self.Fill_projectedImage()
        skimage.io.imsave(filename, img)
    
    def SaveImageTif(self, filename):
        import skimage
        img = self.Fill_projectedImageMasked()
        skimage.io.imsave(filename, img)
import pyproj
class ProjectionImagePyproj(ProjectionImage):
    Transformer_LATLON_XY = pyproj. Transformer.from_crs("EPSG:4326", "EPSG:3857")
    Transformer_XY_LATLON =TRAN_3857_TO_4326 = pyproj. Transformer.from_crs("EPSG:3857","EPSG:4326")    
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
        self.Transformer_LATLON_XY = pyproj. Transformer.from_proj("EPSG:4326", proj)
        self.Transformer_XY_LATLON = pyproj. Transformer.from_proj(proj, "EPSG:4326")