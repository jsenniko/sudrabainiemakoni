import numpy as np
import pandas as pd

from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni.cloudimage import WebMercatorImage
from sudrabainiemakoni import plots, argumentsSM


def doProcessing(args):

    if args.loadProject is None:
        print('Attēls:', args.file)
        cldim = CloudImage(args.id, args.file)
        cldim.setDateFromExif()
        lat, lon = np.array(args.latlon.split(',')).astype('float')
        cldim.setLocation(lat=lat, lon=lon)
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
    else:
        print('Calibrating camera')
        cldim.PrepareCamera(method='optnew')
    if args.saveCamera is not None:
        print('Save camera to:',args.saveCamera)
        cldim.SaveCamera(args.saveCamera)

    if args.saveProject is not None:
        print('Save project to:',args.saveProject)
        cldim.save(args.saveProject)


    if args.plotAltAzGrid is not None:
        print('Plotting AltAz grid')
        plots.PlotAltAzGrid(cldim,  outImageDir = args.plotAltAzGrid, stars = True, showplot=False, from_camera = True)

    if args.reprojectedMap is not None:
        print(f'Preparing reprojected map at height {args.reprojectHeight} km')
        lonmin, lonmax, latmin, latmax, horizontal_resolution_km = np.array(args.webMercParameters.split(',')).astype('float')
        map_lonmin, map_lonmax, map_latmin, map_latmax = np.array(args.mapBounds.split(',')).astype('float')
        webmerc = WebMercatorImage(cldim, lonmin, lonmax, latmin, latmax, horizontal_resolution_km)
        webmerc.prepare_reproject_from_camera(args.reprojectHeight)
        projected_image_hght=webmerc.Fill_projectedImageMasked()
        pp=[[cldim.location.lon.value, cldim.location.lat.value]]
        plots.PlotReferencedImages(webmerc, [projected_image_hght],
                               camera_points=pp,
                               outputFileName=args.reprojectedMap,
                               lonmin=map_lonmin, lonmax=map_lonmax, latmin=map_latmin, latmax=map_latmax,
                               alpha=args.mapAlpha)
if __name__ == "__main__":
    args = argumentsSM.parse_arguments()
    print(args)
    doProcessing(args)

