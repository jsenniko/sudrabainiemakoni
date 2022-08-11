# -*- coding: utf-8 -*-
import os
import simplekml

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


