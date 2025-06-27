# -*- coding: utf-8 -*-
import os
import simplekml
import numpy as np
from sudrabainiemakoni.cloudimage import CloudImage

def mapOverlay(webmerc, projected_image_masked, projHeight, filename, saveimage=True, cloudimage=None):
    #TODO: projected image class
    kml = simplekml.Kml()
    basename = os.path.splitext(filename)[0]
    imagefile = f'{basename}.tif'
    kmlfile = f'{basename}.kml'
    if saveimage:
        import skimage.io
        skimage.io.imsave(imagefile, projected_image_masked)
    
    govr=kml.newgroundoverlay(name='Makoni projicēti')
    govr.latlonbox.west=webmerc.lon_grid.min()
    govr.latlonbox.east=webmerc.lon_grid.max()
    govr.latlonbox.south=webmerc.lat_grid.min()
    govr.latlonbox.north=webmerc.lat_grid.max()
    govr.icon.href = imagefile.replace('\\','/')
    try:
        h=float(projHeight)*1000.0
    except:
        h=80000
    govr.altitude =  h
    govr.altitudemode = 'absolute'
    govr.color='ebffffff' 
    
    if cloudimage is not None and hasattr(cloudimage,'camera'):
        az, el, roll = cloudimage.camera.get_azimuth_elevation_rotation()
        kml.document.camera = simplekml.Camera(longitude=cloudimage.location.lon.value,
                    latitude=cloudimage.location.lat.value,
                    altitude=2,
                    heading=az,
                    tilt=90+el,
                    roll=roll,
                    altitudemode='relativeToGround ' )
        
    
    print('KML', kmlfile)
    kml.save(kmlfile)

def photoToKMLOverlay(kml: simplekml.Kml, cldim: CloudImage, photodist=1000):
    height, width, _ = cldim.imagearray.shape
    photo=kml.newphotooverlay(name=cldim.code)
    photo.icon.href = os.path.abspath(cldim.filename).replace('\\','/')
    #photo.point.coords = [(cldim.location.lon.value,cldim.location.lat.value,cldim.location.height.value)]
    #photo.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/man.png' 
    if cldim.camera.camera_enu.tilt_deg<0:
        heading =180+cldim.camera.camera_enu.heading_deg if cldim.camera.camera_enu.heading_deg<0 else cldim.camera.camera_enu.heading_deg-180
        tilt =-cldim.camera.camera_enu.tilt_deg
        roll =cldim.camera.camera_enu.roll_deg-180 if cldim.camera.camera_enu.roll_deg>0 else 180+cldim.camera.camera_enu.roll_deg
    else:
        heading =cldim.camera.camera_enu.heading_deg
        tilt =cldim.camera.camera_enu.tilt_deg
        roll =cldim.camera.camera_enu.roll_deg
        
    photo.camera = simplekml.Camera(longitude=cldim.location.lon.value, 
                                    latitude=cldim.location.lat.value, 
                                    altitude=cldim.location.height.value, #endcoords_photo_llh[0][2],
                                    altitudemode=simplekml.AltitudeMode.absolute,
                                    heading =heading,
                                    tilt =tilt,
                                    roll =roll,
                                   )
    photo.shape = 'rectangle'
    leftfov = -np.degrees(np.arctan(cldim.camera.camera_enu.center_x_px/cldim.camera.camera_enu.focallength_x_px))
    rightfov = np.degrees(np.arctan((width - cldim.camera.camera_enu.center_x_px)/cldim.camera.camera_enu.focallength_x_px))
    bottomfov = -np.degrees(np.arctan((height-cldim.camera.camera_enu.center_y_px)/cldim.camera.camera_enu.focallength_y_px))
    topfov = np.degrees(np.arctan((cldim.camera.camera_enu.center_y_px)/cldim.camera.camera_enu.focallength_y_px))

    photo.viewvolume = simplekml.ViewVolume(leftfov,rightfov,bottomfov,topfov,photodist)
    return photo
