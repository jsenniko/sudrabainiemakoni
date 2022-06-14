__author__ = 'Juris Seņņikovs'
import skimage
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.coordinates
import astropy.units
import labelatedge
from cloudimage import CloudImage, WebMercatorImage
from calculations import GetImageRaDecGrid
def PlotStars(cloudImage: CloudImage, ax):
    for sr in cloudImage.starReferences:
        ix, iy = sr.pixelcoords
        ax.plot(ix,iy, marker='o', fillstyle='none')
        ax.annotate(sr.name, xy=(ix,iy), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')

def PlotAllStars(cloudImage: CloudImage, outImageDir = None, showplot=True):
    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(cloudImage.imagearray)
    PlotStars(cloudImage, ax)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    if outImageDir is not None:
        fig.savefig(f"{outImageDir}zvaigznes_{cloudImage.code}.jpg", dpi=300, bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.close()
def DrawRADecGrid(ax, coordgrid):
    grid_style={'colors':'white', 'linestyles':'--', 'linewidths':0.5, 'levels':10}
    cs=ax.contour(coordgrid.dec.to_value(), **grid_style)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='left', pad=20, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='right', pad=20, eps=1)
    #ax.clabel(cs, fmt='%.0f')
    cs=ax.contour(coordgrid.ra.wrap_at('180d').to_value(), **grid_style)
    #ra_levels = cs.levels
    fh = r'%.0f$^{\circ}$' #lambda x: astropy.coordinates.Angle(astropy.units.deg * x).to_string(decimal=False, sep='hms',
                            #        unit=astropy.units.hour, fields=1)

    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=fh, side='bottom', pad=-20, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=fh, side='top', pad=-20, eps=1)
def DrawAltAzGrid(ax, aazgrid):
    grid_style={'colors':'#DDFFDD', 'linestyles':'--', 'linewidths':0.5}
    alt_levels=np.arange(-5,65,5)
    if type(aazgrid)==astropy.coordinates.sky_coordinate.SkyCoord:
        alt = aazgrid.alt.to_value()
        az=aazgrid.az.to_value()
    else:
        az = aazgrid[0]
        alt=aazgrid[1]
    az=np.where(az>180, az-360, az)

    cs=ax.contour(alt, **grid_style, levels=alt_levels)
    #ax.clabel(cs, fmt='%.0f', inline=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='left', pad=20, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='right', pad=20, eps=1)
    # apmānam cirkulāro referenci ap ziemeļiem

    az_levels=np.arange(-80,90,10)

    cs=ax.contour(az, **grid_style,  levels=az_levels)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='bottom', pad=-20, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='top', pad=-20, eps=1)

def PlotRADecGrid(cloudImage: CloudImage,  outImageDir = None,  stars = False, showplot=True ):

        fig, ax = plt.subplots(figsize=(20,10))
        ax.imshow(cloudImage.imagearray)
        DrawRADecGrid(ax, cloudImage.radecgrid)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        imid = cloudImage.code
        if stars:
            PlotStars(cloudImage, ax)
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}ekv_coord_{imid}_zvaigznes.jpg', dpi=300, bbox_inches='tight')
        else:
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}ekv_coord_{imid}.jpg', dpi=300, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.close()
def PlotAltAzGrid(cloudImage: CloudImage, outImageDir = None,  stars = False, showplot=True, from_camera = False):
        fig, ax = plt.subplots(figsize=(20,10))
        ax.imshow(cloudImage.imagearray)
        if from_camera and cloudImage.camera.camera_enu is not None:
            DrawAltAzGrid(ax, cloudImage.GetAltAzGrid_fromcamera())
        else:
            DrawAltAzGrid(ax, cloudImage.aazgrid)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        imid = cloudImage.code
        if stars:
            PlotStars(cloudImage, ax)
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}horiz_coord_{imid}_zvaigznes.jpg', dpi=300, bbox_inches='tight')
        else:
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}horiz_coord_{imid}.jpg', dpi=300, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.close()

#def PlotCoordinateGrids(cloudImage: CloudImage, outImageDir = None, showplot = True):
    #coordgrid = GetImageRaDecGrid(cloudImage.imagearray, cloudImage.wcs)
    #PlotRADecGrid(cloudImage, coordgrid, outImageDir,  stars=False, showplot=False)
    #PlotRADecGrid(cloudImage, coordgrid, outImageDir,  stars=True, showplot=showplot)
    #aazgrid = coordgrid.transform_to(im['altaz'])
    #PlotAltAzGrid(imgarr, aazgrid, outImageDir, imid, im, stars=False, showplot=False)
    #PlotAltAzGrid(imgarr, aazgrid, outImageDir, imid, im, stars=True, showplot=showplot)


def DrawEpilineHeightPerKm(px_per_km_grid, cldim2, ax):
    ncolors=20
    cm, norm = matplotlib.colors.from_levels_and_colors(np.linspace(0,ncolors,ncolors+1), matplotlib.cm.viridis(np.linspace(0,1,ncolors)))
    cs=ax.imshow(px_per_km_grid, cmap=cm, norm=norm)
    ax.imshow(cldim2.imagearray, alpha=0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.figure.colorbar(cs)
def PlotEpilineHeightPerKm(px_per_km_grid, cldim2, filename=None):
    fig, ax=plt.subplots(figsize=(20,10))
    DrawEpilineHeightPerKm(px_per_km_grid, cldim2, ax)
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def PlotEpilineGrid(imagearray, epilines, pts = None, filename=None):
    fig, ax=plt.subplots(figsize=(20,10))
    ax.imshow(imagearray)
    for i in range(len(epilines)):
        ax.plot(epilines[i,:,0], epilines[i,:,1],
                color='yellow', marker=None, ms=1, lw=0.8)
    if pts is not None:
        ax.plot(pts[:,0], pts[:,1], marker='o', ls='none', ms=3, mec='red')
    ax.set_xlim(0, imagearray.shape[1])
    ax.set_ylim(imagearray.shape[0],0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
def PlotValidHeightPoints(imagearray, epilines, pts, heightpoints, valid, filename=None):
    fig, ax=plt.subplots(figsize=(20,10))
    ax.imshow(imagearray)
    if valid is None:
        valid = np.zeros(shape=pts.shape[0], dtype='bool')
        valid[:]=True

    for i in range(len(epilines)):
        ax.plot(epilines[i,:,0], epilines[i,:,1],
                color='yellow', marker='o', ms=1, lw=0.8)
    cs=ax.scatter(pts[valid][:,0], pts[valid][:,1], c=heightpoints[valid])
    for pt, h in zip(pts[valid], heightpoints[valid]):
        ax.annotate (f"{h/1000:.0f}km", xy=pt, xytext=pt+np.array([10,-10]) , fontsize=9, color='#AAFFAA')
    ax.set_xlim(0, imagearray.shape[1])
    ax.set_ylim(imagearray.shape[0],0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    fig.colorbar(cs)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def PlotReferencedImages(webmerc: WebMercatorImage,
                         projected_images,
                         camera_points=[],
                         outputFileName = None, showplot = False,
                         lonmin=15, lonmax=30, latmin=56, latmax=62,
                         alpha=0.8):
    import tilemapbase
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()
    extent = tilemapbase.Extent.from_lonlat(lonmin,lonmax,latmin,latmax)
    e1=tilemapbase.Extent.from_3857(webmerc.xmin,webmerc.xmax, webmerc.ymax, webmerc.ymin)
    e1 = e1.to_project_web_mercator()
    import matplotlib.transforms
    w=16
    h=9*w/16
    hbb=7
    fig, ax = plt.subplots(figsize=(w,h), facecolor='#FAFAFA')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tilemapbase.Plotter(extent, t, width=500)
    plotter.plot(ax, t)
    for projected_image in projected_images:
        ax.imshow(projected_image, extent=(e1.xmin, e1.xmax, e1.ymax, e1.ymin), alpha=alpha)

    for plonlat in camera_points:
        p = tilemapbase.project(plonlat[0], plonlat[1])
        ax.plot(p[0],p[1],marker='o', ms=12)
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=300, bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.close()