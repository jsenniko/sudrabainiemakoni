# -*- coding: utf-8 -*-
import sys
import os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni import plots
from projection import ProjectionImagePyproj
import exifread
import pytz
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
# OSM tiles
import cartopy.io.img_tiles as cimgt

request = cimgt.OSM() 
# TODO arbitrary projections
mapproj='+proj=eqdc +lon_0=24 +lat_1=58 +lat_2=61 +lat_0=59.5 +datum=WGS84 +units=m +no_defs'
cartopy_proj=cartopy.crs.EquidistantConic(standard_parallels=(58, 61), central_longitude=24,central_latitude=59.5)

# TODO arbitrary time-zones
timezone = pytz.timezone('Europe/Riga')
timezone_UTC = pytz.timezone('UTC')
#lonmin, lonmax, latmin, latmax, horizontal_resolution_km = 14.0,34.0,56.0,62.0,0.25
proj_bounds = 14.0,34.0,56.0,62.0,0.25
#map_lonmin, map_lonmax, map_latmin, map_latmax = 14.0,34.0,55.5,62.0
map_bounds = 14.0,34.0,55.5,62.0
dpi=300

def get_exifdate(fn):
    f = open(fn, 'rb')
    tags = exifread.process_file(f)
    x=tags['EXIF DateTimeOriginal']
    exifdate =datetime.datetime.strptime(x.values, '%Y:%m:%d %H:%M:%S')
    f.close()
    return exifdate
def init_projection_map(cldim, height_km, mapproj=mapproj, proj_bounds =proj_bounds):
    projectionIm=ProjectionImagePyproj(mapproj, cldim, *proj_bounds)
    projectionIm.prepare_reproject_from_camera(height_km)
    return projectionIm
def getProjectedImage_frame(projectionIm, filename):
    cldim = projectionIm.cloudImage
    cldim.filename=filename
    cldim.setDateFromExif()
    cldim.LoadImage(reload=True)    
    projected_image_2=projectionIm.Fill_projectedImageMasked()
    return projected_image_2
    
    




def init_plot(projectionIm, map_bounds =map_bounds, height=2160, aspect=3/2, alpha=0.85, dpi=dpi,
              plot_latlon_gridlines=False,
              plot_xy_gridlines=True,
              plot_date_label=True,
              cartopy_proj=cartopy_proj,
              osm_level=7):
    cldim = projectionIm.cloudImage
    projected_image_2=getProjectedImage_frame(projectionIm, cldim.filename)    
    
    width = height * aspect
    fig=plt.figure(figsize=(width/dpi,height/dpi))
    # TODO cartopy proj from proj
    ax=fig.add_axes((0,0,1,1),projection=cartopy_proj)
    ax.set_extent(map_bounds,  crs= ccrs.PlateCarree())   
    ax.add_feature(cf.COASTLINE, lw=1, ls='--')
    ax.add_feature(cf.BORDERS, lw=1,color='black', ls='--')
    pic_imshow=ax.imshow(projected_image_2, extent=(projectionIm.xmin, projectionIm.xmax, projectionIm.ymin, projectionIm.ymax), 
                         alpha=alpha)
    # https://github.com/SciTools/cartopy/issues/1048#issuecomment-417001744
    ax.add_image(request, osm_level, zorder=-10,interpolation='spline36', regrid_shape=height)
    if plot_latlon_gridlines:
        gl=ax.gridlines(draw_labels=True,ls='--')
        # TODO locator parameters
        gl.xlocator=matplotlib.ticker.FixedLocator(np.arange(12,36,2))
        gl.ylocator=matplotlib.ticker.FixedLocator(np.arange(54,63,1))    
    if plot_xy_gridlines:
        # TODO locator parameters
        dx=50000
        ax.set_xticks(np.arange(-12*dx,dx*12,dx))
        ax.set_yticks(np.arange(-9*dx,dx*7,dx))
        ax.set_xlim(-10*dx, 10*dx)
        ax.set_ylim(-8*dx, 6*dx)
        ax.tick_params(labelbottom=False, labelleft=False, length=0) 
        ax.grid(ls='--', color='black', lw=0.5)    
        # scale arrow
        ax.annotate('', xy=(-8*dx,-6.75*dx), xytext=(-6*dx,-6.75*dx), arrowprops=dict(arrowstyle='<->', lw=2))
        ax.annotate(f'{2*dx/1000:.0f} km', xy=(-7*dx,-6.75*dx), xytext=(0,2), textcoords='offset points', ha='center', va='bottom', fontsize=14)    
    date_label=None
    if plot_date_label:
        date_localized=timezone_UTC.localize(cldim.date.to_datetime())
        ds=date_localized.astimezone(timezone)
        date_label = ax.annotate(ds.isoformat(), xy=(0,0), xytext=(5,5), xycoords='axes fraction', 
                    textcoords='offset points', 
                    bbox=dict(facecolor='white', edgecolor='black'),
                    ha='left', va='bottom', fontsize=14)
    return ax, pic_imshow, date_label
 



def update_frame(framefilename, projectionIm, date_label, pic_imshow):
    cldim = projectionIm.cloudImage
    projected_image_2=getProjectedImage_frame(projectionIm, framefilename)   
    date_localized=timezone_UTC.localize(cldim.date.to_datetime())
    ds=date_localized.astimezone(timezone)
    date_label.set_text(ds.isoformat())
    pic_imshow.set_array(projected_image_2)

   
def prepare_animation_frames(source_project, height_km, 
            source_files, output_dir,
            mapproj=mapproj, proj_bounds=proj_bounds, map_bounds=map_bounds):
    os.makedirs(output_dir, exist_ok=True)
    cldim=CloudImage.load(source_project) 
    projectionIm=init_projection_map(cldim, height_km, mapproj,  proj_bounds)
    ax, pic_imshow, date_label = init_plot(projectionIm, map_bounds, dpi=dpi)
    for framenr, filename in enumerate(source_files):
        update_frame(filename, projectionIm, date_label, pic_imshow)
        ax.figure.savefig(f'{output_dir}/karte_{(framenr+1):05d}.jpg', dpi=dpi)

def prepare_funcanimation(source_project, height_km, 
            source_files,
            mapproj=mapproj, proj_bounds=proj_bounds, map_bounds=map_bounds,
            interval=20):
    cldim=CloudImage.load(source_project) 
    projectionIm=init_projection_map(cldim, height_km, mapproj,  proj_bounds)
    ax, pic_imshow, date_label = init_plot(projectionIm, map_bounds, dpi=dpi)
    from functools import partial
    anim = matplotlib.animation.FuncAnimation(fig=ax.figure, func=partial(update_frame, 
                                                                    projectionIm=projectionIm,
                                                                    date_label=date_label,
                                                                    pic_imshow=pic_imshow), 
                                         frames=source_files, interval=interval, repeat=False)
    return anim

def save_standalone_image(source_project, height_km, output_pic, source_file=None):
    cldim=CloudImage.load(source_project)
    if source_file is not None:
        cldim.filename=source_file
        cldim.LoadImage(reload=True)
        cldim.setDateFromExif()

    projectionIm=init_projection_map(cldim, height_km)
    ax, pic_imshow, date_label = init_plot(projectionIm,
                  plot_latlon_gridlines=True,
                  plot_xy_gridlines=True)
    ax.figure.savefig(output_pic, dpi=300, bbox_inches='tight')

if __name__=="__main__":
    source_dir=r'test_timelapse'
    source_project = 'test.proj'
    source_frames = range(1,489+1,1)
    source_files = [f'{source_dir}/{framenr:05d}.jpg' for framenr in source_frames]
    output_dir='map_frames'
    height_km = 80   
    prepare_animation_frames(source_project, height_km, source_files, output_dir)
    #ani = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=source_frames, interval=20, repeat=False)

    