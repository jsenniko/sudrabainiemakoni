__author__ = 'Juris Seņņikovs'
import os
import astropy
import astropy.coordinates
import astropy.units as u
import datetime, pytz
import numpy as np
import pandas as pd
import skimage
import skimage.io
import cameratransform as ct
import pymap3d
from sudrabainiemakoni import utils
from sudrabainiemakoni import geoutils
from sudrabainiemakoni import calculations
from sudrabainiemakoni.starreference import StarReference
from sudrabainiemakoni.cloudimage_camera import Camera
import cameraprojections


# Import the new Web Mercator implementation and create backwards compatibility alias
from .webmercatorimage import ProjectionImageWebMercator
WebMercatorImage = ProjectionImageWebMercator

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




class CloudImage:
    timezone = pytz.timezone('Europe/Riga')
    @classmethod
    def load(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        if not os.path.exists(d.filename):
            # try relative path
            test = os.path.join(os.path.dirname(filename), os.path.basename(d.filename))
            print(f'Trying 1 {test}')
            if os.path.exists(test):
                d.filename = test
            
            fn = d.filename
            t1=''
            while fn != '':
                fn,f2 = os.path.split(fn)
                if t1=='':
                    t1 = f2
                else:
                    t1 = os.path.join(f2, t1)
                test = os.path.join(os.path.dirname(filename), t1)
                print(f'Trying 2 {test}')
                if os.path.exists(test):
                    d.filename = test
            print(f'Image does not exist {d.filename}')
        #d.LoadCamera(filename)
        
        # Fix camera reference after pickle loading
        if hasattr(d, 'camera') and d.camera is not None:
            d.camera.cloudImage = d
        
        return d

    def __init__(self, code, filename):
        self.code = code
        self.starReferences: list[StarReference] = []
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
    def from_files(cls, case_id, filename_jpg, filename_stars, lat, lon, height=0):
        cldim = CloudImage(case_id, filename_jpg)
        cldim.setDateFromExif()
        if lat is not None and lon is not None:
            cldim.setLocation(lat=lat, lon=lon, height=height)
        else:
            cldim.setLocationExif()
        print('UTC:', cldim.date)
        print(cldim.location.to_geodetic())
        # uzstādām zvaigžņu sarakstu
        cldim.loadStarReferences(filename_stars)
        # izdrukājam zvaigžņu ekvatoriālās un pikseļu koordinātes pārbaudes nolūkos
        print(cldim.getSkyCoords())
        print(cldim.getPixelCoords())               
        return cldim    
    def __getstate__(self):
        state = self.__dict__
        invalid = {"xxx"}
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
    def getLocation(self):        
        return self.location.lat.value, self.location.lon.value, self.location.height.value
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
    
    def addStarWithAltAz(self, name, pixelcoords, az_deg, alt_deg):
        """
        Add a star reference with Alt-Az coordinates.
        
        Args:
            name: Star name or identifier
            pixelcoords: [x, y] pixel coordinates in image
            az_deg: Azimuth in degrees
            alt_deg: Altitude in degrees
        """
        altaz_coord = astropy.coordinates.AltAz(
            az=az_deg * u.deg,
            alt=alt_deg * u.deg
        )
        star = StarReference(name, pixelcoords, altaz_coord)
        self.starReferences.append(star)
        return star
    
    def saveStarReferences(self, filename):
        data = {
            'name': [r.name for r in self.starReferences],
            'ix': [r.pixelcoords[0] for r in self.starReferences],
            'iy': [r.pixelcoords[1] for r in self.starReferences]
        }
        
        # Check if any stars have Alt-Az coordinates
        has_altaz = any(r.hasDirectAltAz() for r in self.starReferences)
        
        if has_altaz:
            # Include Alt-Az coordinates and source information
            data['az'] = [r.altaz_coord.az.deg if r.hasDirectAltAz() else np.nan 
                         for r in self.starReferences]
            data['alt'] = [r.altaz_coord.alt.deg if r.hasDirectAltAz() else np.nan 
                          for r in self.starReferences]
            data['coord_source'] = ['altaz' if r.hasDirectAltAz() else 'name' 
                                   for r in self.starReferences]
            # Save with headers for new format
            df = pd.DataFrame(data)
            df.to_csv(filename, sep='\t', header=True, index=False)
        else:
            # Save in old format for backward compatibility
            df = pd.DataFrame(data)
            df.to_csv(filename, sep='\t', header=None, index=False)
    
    def loadStarReferences(self, filename):
        """
        Load star references from file, supporting both old and new formats.
        
        Old format: tab-separated file with no headers (name, ix, iy)
        New format: tab-separated file with headers including Alt-Az coordinates
        """
        try:
            # First try to read with headers (new format)
            df = pd.read_csv(filename, sep='\t')
            
            if 'name' in df.columns:
                # New format with headers
                self.starReferences = []
                for _, row in df.iterrows():
                    altaz_coord = None
                    
                    # Check if we have Alt-Az coordinates
                    if 'az' in row and 'alt' in row and not pd.isna(row['az']) and not pd.isna(row['alt']):
                        altaz_coord = astropy.coordinates.AltAz(
                            az=row['az'] * u.deg,
                            alt=row['alt'] * u.deg
                        )
                    
                    star = StarReference(row['name'], [row['ix'], row['iy']], altaz_coord)
                    self.starReferences.append(star)
                
                # Resolve RA/DEC coordinates for stars that don't have Alt-Az
                for star in self.starReferences:
                    if not star.hasDirectAltAz():
                        star.getSkyCoord()
            else:
                # Old format without headers - assume columns are name, ix, iy
                starnames = df[0]
                pixels = np.array(df[[1,2]])
                self.setStarReferences(starnames, pixels)
                
        except Exception as e:
            # Fallback: try old format without headers
            try:
                df = pd.read_csv(filename, sep='\t', header=None)
                starnames = df[0]
                pixels = np.array(df[[1,2]])
                self.setStarReferences(starnames, pixels)
            except:
                # Empty star list if file can't be read
                self.starReferences = []
                print(f"Could not load star references from {filename}: {e}")
        
    def getPixelCoords(self):
        return np.array([r.pixelcoords for r in self.starReferences])
    def getSkyCoords(self):
        try:
            return astropy.coordinates.SkyCoord([r.skycoord for r in self.starReferences])
        except:
            return []


    def GetAltAzGrid_fromcamera(self):
        return Camera.GetAltAzGrid_fromcamera(self.imagearray.shape[1], self.imagearray.shape[0], self.camera.camera_enu)


    def PrepareCamera(self, **params):
        self.camera = Camera(self)
        self.camera.Fit(**params)
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
        enu_coords = []
        
        for star in self.starReferences:
            # Try to get ENU coordinates from each star
            enu = star.getENUUnitVector(self.altaz)
            if enu is not None:
                enu_coords.append(enu)
            else:
                print(f'WARNING: star without coordinates {star}')
        
        if enu_coords:
            return np.array(enu_coords)
        else:
            return np.array([])
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

