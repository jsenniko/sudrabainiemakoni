import os, glob
import numpy as np
import pandas as pd

from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni.cloudimage import WebMercatorImage
from sudrabainiemakoni import plots, argumentsSM

def doProcessing(args):

    cldim = CloudImage.load(args.loadProject)

    print(f'Preparing reprojected map at height {args.reprojectHeight} km')
    lonmin, lonmax, latmin, latmax, horizontal_resolution_km = np.array(args.webMercParameters.split(',')).astype('float')
    map_lonmin, map_lonmax, map_latmin, map_latmax = np.array(args.mapBounds.split(',')).astype('float')
    webmerc = WebMercatorImage(cldim, lonmin, lonmax, latmin, latmax, horizontal_resolution_km)
    webmerc.prepare_reproject_from_camera(args.reprojectHeight)


    print('Directory of input images ', args.timelapseInputDir)
    inputFiles = list(sorted(glob.glob(f'{args.timelapseInputDir}/*.jpg')))

    geodir=f'{args.timelapseOutputDir}/georeferenced'
    if not os.path.exists(geodir):
        os.makedirs(geodir)
    if args.prepareGeoreferenced:
        print('Preparing georeferenced images to ',geodir)
        for i, fn in enumerate(inputFiles):
            print('Processing',os.path.split(fn)[1])
            cldim.filename=fn
            cldim.LoadImage(reload=True)
            projected_image=webmerc.Fill_projectedImageMasked()
            np.save(f'{geodir}/{i:03d}.npy', projected_image)
    if args.prepareMaps:
        # karšu animācija
        mapdir = f'{args.timelapseOutputDir}/maps'
        if not os.path.exists(mapdir):
            os.makedirs(mapdir)
        map_lonmin, map_lonmax, map_latmin, map_latmax = np.array(args.mapBounds.split(',')).astype('float')
        georeferencedFiles = list(sorted(glob.glob(f'{geodir}/*.npy')))

        plotInitData = plots.InitPlotReferencedImages(webmerc, map_lonmin, map_lonmax, map_latmin, map_latmax)

        for i, fn in enumerate(georeferencedFiles):
            fnout = f'{mapdir}/{i:03d}.jpg'
            print('Preparing map file:',os.path.split(fnout)[1])
            projected_image = np.load(fn)
            plots.PlotReferencedImages(webmerc, [projected_image],
                               camera_points=[],
                               outputFileName=fnout,
                               initData = plotInitData,
                               alpha=0.85)


if __name__ == "__main__":
    args = argumentsSM.parse_arguments_timelapse()
    print(args)
    doProcessing(args)



