import os
import numpy as np
import pandas as pd
import skimage
import skimage.io

from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni.cloudimage import WebMercatorImage
from sudrabainiemakoni import plots, argumentsSM


def doProcessing(args):

    if args.loadProject is None:
        print('Attēls:', args.file)
        cldim = CloudImage(args.id, args.file)
        cldim.setDateFromExif()
        if args.latlon is not None:
            lat, lon = np.array(args.latlon.split(',')).astype('float')
            cldim.setLocation(lat=lat, lon=lon)
        else:
            cldim.setLocationExif()
        if args.zvaigznes is not None:
            df = pd.read_csv(args.zvaigznes, sep='\t', header=None)
            starnames = df[0]
            pixels=np.array(df[[1,2]])
            cldim.setStarReferences(starnames, pixels)
            cldim.GetWCS(sip_degree=2, fit_parameters={'projection':'TAN'})
    else:
        cldim = CloudImage.load(args.loadProject)
    if args.plotRAGrid is not None:
        print('Plotting RA grid')
        plots.PlotRADecGrid(cldim, outImageDir = args.plotRAGrid,  stars = False, showplot=False )
    if args.loadCamera is not None:
        print('Load camera from:',args.loadCamera)
        cldim.LoadCamera(args.loadCamera)
    elif args.loadProject is None:
        print('Calibrating camera')
        cldim.PrepareCamera(distortion=args.optimizeDistortion, centers=args.notOptimizeCenter, separate_x_y=args.notOptimizeUnsymmetric)
    if args.saveCamera is not None:
        print('Save camera to:',args.saveCamera)
        cldim.SaveCamera(args.saveCamera)

    if args.saveProject is not None:
        print('Save project to:',args.saveProject)
        cldim.save(args.saveProject)


    if args.plotAltAzGrid is not None:
        print('Plotting AltAz grid')
        plots.PlotAltAzGrid(cldim,  outImageDir = args.plotAltAzGrid, stars = True, showplot=False, from_camera = True)

    if args.reprojectedMap is not None or args.reprojectedImage is not None:
        lonmin, lonmax, latmin, latmax, horizontal_resolution_km = np.array(args.webMercParameters.split(',')).astype('float')
        if args.reprojectedMap:
            pp=[[cldim.location.lon.value, cldim.location.lat.value]]
            map_lonmin, map_lonmax, map_latmin, map_latmax = np.array(args.mapBounds.split(',')).astype('float')
        webmerc = WebMercatorImage(cldim, lonmin, lonmax, latmin, latmax, horizontal_resolution_km)

        height_start = args.reprojectHeight[0]
        height_end = height_start
        height_step = 1.0
        if len(args.reprojectHeight)>=3:
            height_end = args.reprojectHeight[1]
            height_step = args.reprojectHeight[2]
        for height in np.arange(height_start, height_end+height_step, height_step):
            print(f'Preparing reprojected map at height {height} km')
            webmerc.prepare_reproject_from_camera(height)
            projected_image_hght=webmerc.Fill_projectedImageMasked()
            if args.reprojectedImage:
                fnimage=os.path.splitext(args.reprojectedImage)[0]+f'_{height}'
                if args.reprojectedImageFormat=='tif':
                    jgwext='.tfw'
                    skimage.io.imsave(f'{fnimage}.tif', projected_image_hght)
                else:
                    jgwext='.jgw'
                    projected_image_jpg =   projected_image_hght[:,:,0:3]
                    skimage.io.imsave(f'{fnimage}.jpg', projected_image_jpg)
                if args.reprojectedImageJGW:
                     webmerc.SaveJgw(f'{fnimage}{jgwext}')


            if args.reprojectedMap:
                fnimage=os.path.splitext(args.reprojectedMap)[0]+f'_{height}.jpg'
                plots.PlotReferencedImages(webmerc, [projected_image_hght],
                                   camera_points=pp,
                                   outputFileName=fnimage,
                                   lonmin=map_lonmin, lonmax=map_lonmax, latmin=map_latmin, latmax=map_latmax,
                                   alpha=args.mapAlpha)
if __name__ == "__main__":
    args = argumentsSM.parse_arguments()
    print(args)
    doProcessing(args)

