__author__ = 'Juris Seņņikovs'
import skimage
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.coordinates
import astropy.units
from sudrabainiemakoni import labelatedge
from sudrabainiemakoni.cloudimage import CloudImage, WebMercatorImage
from sudrabainiemakoni.calculations import GetImageRaDecGrid
def PlotStars(cloudImage: CloudImage, ax, show_circles=True, show_names=True):
    """
    Plot star references on the axes.

    Args:
        cloudImage: CloudImage object containing star references
        ax: matplotlib axes
        show_circles: Show star position circles (default True)
        show_names: Show star name labels (default True)
    """
    for sr in cloudImage.starReferences:
        ix, iy = sr.pixelcoords
        if show_circles:
            ax.plot(ix, iy, marker='o', fillstyle='none')
        if show_names:
            ax.annotate(sr.name, xy=(ix, iy), xytext=(3, 3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')

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
def DrawAltAzGrid(ax, aazgrid, azlevels=None, altlevels=None, nticks=15):
    locator=matplotlib.ticker.MaxNLocator(nticks, steps=[1,2,5,10], prune ='both')
    #locator_alt=matplotlib.ticker.MaxNLocator(nticks//2, steps=[1,2,5,10], prune ='both')
    grid_style={'colors':'#DDFFDD', 'linestyles':'--', 'linewidths':0.5}
    if type(aazgrid)==astropy.coordinates.sky_coordinate.SkyCoord:
        alt = aazgrid.alt.to_value()
        az=aazgrid.az.to_value()
    else:
        az = aazgrid[0]
        alt=aazgrid[1]

    alt_min, alt_max=alt.min(), alt.max()
    alt_levels=locator.tick_values(vmin=alt_min, vmax=alt_max) if altlevels is None else altlevels #np.arange(-5,65,5)
    az_min, az_max=az.min(), az.max()
    if az_max-az_min>180:
        az=np.where(az>180, az-360, az)

    cs=ax.contour(alt, **grid_style, levels=alt_levels)
    #ax.clabel(cs, fmt='%.0f', inline=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='left', pad=20)#, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=r'%.0f$^{\circ}$', side='right', pad=20)#, eps=1)
    # apmānam cirkulāro referenci ap ziemeļiem
    az_levels=locator.tick_values(vmin=az_min, vmax=az_max) if azlevels is None else azlevels#  np.arange(-80,90,10)

    cs=ax.contour(az, **grid_style,  levels=az_levels)
    fmt = lambda x: r'{0:.0f}$^{{\circ}}$'.format(x if x>=0 else x+360)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=fmt, side='bottom', pad=-20)#, eps=1)
    labelatedge.labelAtEdge(cs.levels, cs, ax, fmt=fmt, side='top', pad=-20)#, eps=1)

def DrawAltAzGrid_v2(ax, aazgrid,
                     azlevels=None, altlevels=None,
                     nticks=15, nticks_major=None,
                     minor_az_step=None, major_az_step=None,
                     minor_alt_step=None, major_alt_step=None,
                     minor_grid_style=None, major_grid_style=None,
                     label_major_only=True, label_placement='outside',
                     label_side='all', label_pad=10, label_fmt=None,
                     label_kwargs=None,
                     grid_is_buffered=True):
    """
    Draw altitude/azimuth grid with separate major and minor gridlines.

    Args:
        ax: matplotlib axes
        aazgrid: Either SkyCoord with alt/az or tuple of (az, alt) arrays

    Keyword Args:
        azlevels: Explicit azimuth levels (overrides automatic calculation)
        altlevels: Explicit altitude levels (overrides automatic calculation)
        nticks (int): Maximum number of minor gridlines (for automatic mode)
        nticks_major (int): Maximum number of major gridlines (for automatic mode).
                           If None, derived from nticks
        minor_az_step (float): Step for minor azimuth gridlines in degrees (manual mode)
        major_az_step (float): Step for major azimuth gridlines in degrees (manual mode)
        minor_alt_step (float): Step for minor altitude gridlines in degrees (manual mode)
        major_alt_step (float): Step for major altitude gridlines in degrees (manual mode)
        minor_grid_style (dict): Style for minor gridlines
        major_grid_style (dict): Style for major gridlines
        label_major_only (bool): If True, only label major gridlines
        label_placement (str): 'outside' or 'inside'
        label_side (str): 'left', 'right', 'top', 'bottom', or 'all'
        label_pad (float): Padding for labels from edge
        label_fmt (str or dict): Format for labels
        label_kwargs (dict): Additional text properties for labels
    """
    if type(aazgrid) == astropy.coordinates.sky_coordinate.SkyCoord:
        alt = aazgrid.alt.to_value()
        az = aazgrid.az.to_value()
    else:
        az = aazgrid[0]
        alt = aazgrid[1]
    if grid_is_buffered:
        grid_ix = (-1,az.shape[1]-1)
        grid_iy = (-1,az.shape[0]-1)
    else:
        grid_ix = (0,az.shape[1])
        grid_iy = (0,az.shape[0])
    if minor_grid_style is None:
        minor_grid_style = {'colors': '#DDFFDD', 'linestyles': ':', 'linewidths': 0.3}
    if major_grid_style is None:
        major_grid_style = {'colors': '#DDFFDD', 'linestyles': '--', 'linewidths': 0.8}

    if label_kwargs is None:
        label_kwargs = {}

    # Use brighter color for inside labels to be visible on dark backgrounds
    if label_placement == 'inside' and 'color' not in label_kwargs:
        label_kwargs = label_kwargs.copy()
        label_kwargs['color'] = 'white'
        label_kwargs['fontweight'] = 'bold'

    if label_fmt is None:
        alt_fmt = r'%.0f$^{\circ}$'
        az_fmt = lambda x: r'{0:.0f}$^{{\circ}}$'.format(x if x >= 0 else x + 360)
    elif isinstance(label_fmt, dict):
        alt_fmt = label_fmt.get('alt', r'%.0f$^{\circ}$')
        az_fmt = label_fmt.get('az', lambda x: r'{0:.0f}$^{{\circ}}$'.format(x if x >= 0 else x + 360))
    else:
        alt_fmt = label_fmt
        az_fmt = label_fmt

    alt_min, alt_max = alt.min(), alt.max()
    az_min, az_max = az.min(), az.max()

    if az_max - az_min > 180:
        az = np.where(az > 180, az - 360, az)
        az_min, az_max = az.min(), az.max()

    if altlevels is None:
        if minor_alt_step is not None:
            alt_minor_levels = np.arange(
                np.floor(alt_min / minor_alt_step) * minor_alt_step,
                np.ceil(alt_max / minor_alt_step) * minor_alt_step + minor_alt_step,
                minor_alt_step
            )
        else:
            locator = matplotlib.ticker.MaxNLocator(nticks, steps=[1, 2, 5, 10], prune='both')
            alt_minor_levels = locator.tick_values(vmin=alt_min, vmax=alt_max)

        if major_alt_step is not None:
            alt_major_levels = np.arange(
                np.floor(alt_min / major_alt_step) * major_alt_step,
                np.ceil(alt_max / major_alt_step) * major_alt_step + major_alt_step,
                major_alt_step
            )
        else:
            if nticks_major is None:
                nticks_major = max(3, nticks // 3)
            locator_major = matplotlib.ticker.MaxNLocator(nticks_major, steps=[1, 2, 5, 10], prune='both')
            alt_major_levels = locator_major.tick_values(vmin=alt_min, vmax=alt_max)
    else:
        alt_minor_levels = np.array(altlevels)
        alt_major_levels = np.array(altlevels)

    if azlevels is None:
        if minor_az_step is not None:
            az_minor_levels = np.arange(
                np.floor(az_min / minor_az_step) * minor_az_step,
                np.ceil(az_max / minor_az_step) * minor_az_step + minor_az_step,
                minor_az_step
            )
        else:
            locator = matplotlib.ticker.MaxNLocator(nticks, steps=[1, 2, 5, 10], prune='both')
            az_minor_levels = locator.tick_values(vmin=az_min, vmax=az_max)

        if major_az_step is not None:
            az_major_levels = np.arange(
                np.floor(az_min / major_az_step) * major_az_step,
                np.ceil(az_max / major_az_step) * major_az_step + major_az_step,
                major_az_step
            )
        else:
            if nticks_major is None:
                nticks_major = max(3, nticks // 3)
            locator_major = matplotlib.ticker.MaxNLocator(nticks_major, steps=[1, 2, 5, 10], prune='both')
            az_major_levels = locator_major.tick_values(vmin=az_min, vmax=az_max)
    else:
        az_minor_levels = np.array(azlevels)
        az_major_levels = np.array(azlevels)

    alt_minor_only = np.setdiff1d(alt_minor_levels, alt_major_levels)
    az_minor_only = np.setdiff1d(az_minor_levels, az_major_levels)

    # Determine which sides to label for altitude (left/right) and azimuth (top/bottom)
    if label_side == 'all':
        alt_label_side = 'left,right'
        az_label_side = 'top,bottom'
    elif label_side in ['left', 'right']:
        alt_label_side = label_side
        az_label_side = None
    elif label_side in ['top', 'bottom']:
        alt_label_side = None
        az_label_side = label_side
    else:
        alt_label_side = 'left,right'
        az_label_side = 'top,bottom'

    if len(alt_minor_only) > 0:
        cs_alt_minor = ax.contour(np.arange(*grid_ix), np.arange(*grid_iy), alt, **minor_grid_style, levels=alt_minor_only)
        if not label_major_only and alt_label_side is not None:
            for side in alt_label_side.split(','):
                labelatedge.labelAtEdge_v2(cs_alt_minor.levels, cs_alt_minor, ax,
                                           fmt=alt_fmt,
                                           side=side,
                                           pad=label_pad,
                                           label_levels=None,
                                           placement=label_placement,
                                           **label_kwargs)

    if len(alt_major_levels) > 0:
        cs_alt_major = ax.contour(np.arange(*grid_ix), np.arange(*grid_iy), alt, **major_grid_style, levels=alt_major_levels)
        if alt_label_side is not None:
            for side in alt_label_side.split(','):
                labelatedge.labelAtEdge_v2(cs_alt_major.levels, cs_alt_major, ax,
                                           fmt=alt_fmt,
                                           side=side,
                                           pad=label_pad,
                                           label_levels=None,
                                           placement=label_placement,
                                           **label_kwargs)

    if len(az_minor_only) > 0:
        cs_az_minor = ax.contour(np.arange(*grid_ix), np.arange(*grid_iy), az, **minor_grid_style, levels=az_minor_only)
        if not label_major_only and az_label_side is not None:
            for side in az_label_side.split(','):
                labelatedge.labelAtEdge_v2(cs_az_minor.levels, cs_az_minor, ax,
                                           fmt=az_fmt,
                                           side=side,
                                           pad=label_pad,
                                           label_levels=None,
                                           placement=label_placement,
                                           **label_kwargs)

    if len(az_major_levels) > 0:
        cs_az_major = ax.contour(np.arange(*grid_ix), np.arange(*grid_iy),az, **major_grid_style, levels=az_major_levels)
        if az_label_side is not None:
            for side in az_label_side.split(','):
                labelatedge.labelAtEdge_v2(cs_az_major.levels, cs_az_major, ax,
                                           fmt=az_fmt,
                                           side=side,
                                           pad=label_pad,
                                           label_levels=None,
                                           placement=label_placement,
                                           **label_kwargs)


def PlotRADecGrid(cloudImage: CloudImage,  outImageDir = None,  stars = False, showplot=True ):

        fig, ax = plt.subplots(figsize=(20,10))
        ax.imshow(cloudImage.imagearray)
        from sudrabainiemakoni.wcs_coordinate_systems import WCSCoordinateSystemsAdapter
        wcs_adapter = WCSCoordinateSystemsAdapter(cloudImage)
        DrawRADecGrid(ax, wcs_adapter.radecgrid)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        imid = cloudImage.code
        if stars:
            PlotStars(cloudImage, ax)
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}/ekv_coord_{imid}_zvaigznes.jpg', dpi=300, bbox_inches='tight')
        else:
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}/ekv_coord_{imid}.jpg', dpi=300, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.close()
def PlotAltAzGrid_v2(cloudImage: CloudImage, outImageDir=None, stars=False, showplot=True,
                     from_camera=True, ax=None, grid_kwargs=None, exact_size=False):
    """
    Plot altitude/azimuth grid with improved styling options.

    Args:
        cloudImage: CloudImage object
        outImageDir: Directory to save output image
        stars: Whether to plot star references
        showplot: Whether to display the plot
        from_camera: Use camera-based grid calculation
        ax: Existing axes to plot on (if None, creates new figure)
        grid_kwargs: Dictionary of parameters to pass to DrawAltAzGrid_v2
        exact_size: If True, create figure matching exact image dimensions for overlay
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    if exact_size:
        h_px, w_px = cloudImage.imagearray.shape[:2]
        dpi = grid_kwargs.pop('dpi', 100)
        fig, ax = plt.subplots(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        ax.set_position([0, 0, 1, 1])
        doPlot = True
    elif ax is None:
        doPlot = True
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        doPlot = False
        fig = ax.figure

    ax.imshow(cloudImage.imagearray)

    if from_camera and cloudImage.camera.camera_enu is not None:
        DrawAltAzGrid_v2(ax, cloudImage.GetAltAzGrid_fromcamera(buffer=True), **grid_kwargs)
    else:
        from sudrabainiemakoni.wcs_coordinate_systems import WCSCoordinateSystemsAdapter
        wcs_adapter = WCSCoordinateSystemsAdapter(cloudImage)
        DrawAltAzGrid_v2(ax, wcs_adapter.aazgrid, **grid_kwargs)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    imid = cloudImage.code
    if outImageDir is not None:
        if stars:
            PlotStars(cloudImage, ax)
            fig.savefig(f'{outImageDir}/horiz_coord_{imid}_zvaigznes_v2.jpg',
                        dpi=300, bbox_inches='tight' if not exact_size else None,
                        pad_inches=0 if exact_size else 0.1)
        else:
            fig.savefig(f'{outImageDir}/horiz_coord_{imid}_v2.jpg',
                        dpi=300, bbox_inches='tight' if not exact_size else None,
                        pad_inches=0 if exact_size else 0.1)

    if doPlot:
        if showplot:
            plt.show()
        else:
            plt.close()


def CreateGridOverlayFigure(cloudImage: CloudImage, grid_kwargs=None, dpi=100,
                             filename=None, close_figure=False):
    """
    Create a figure with exact image dimensions and grid overlay.

    Args:
        cloudImage: CloudImage object with camera calibration
        grid_kwargs: Dictionary of parameters to pass to DrawAltAzGrid_v2
        dpi: DPI for the figure (default 100)
        filename: If provided, save the figure to this file
        close_figure: If True, close the figure after saving (default False)

    Returns:
        Tuple of (fig, ax) - matplotlib figure and axes objects
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    # Get image dimensions
    h_px, w_px = cloudImage.imagearray.shape[:2]

    # Create figure with exact image dimensions
    fig, ax = plt.subplots(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

    # Remove all margins and padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')

    # Draw image
    ax.imshow(cloudImage.imagearray, aspect='auto', interpolation='none')

    # Force inside placement for overlay
    grid_kwargs = grid_kwargs.copy()
    grid_kwargs['label_placement'] = 'inside'

    # Draw grid
    from sudrabainiemakoni.wcs_coordinate_systems import WCSCoordinateSystemsAdapter
    if hasattr(cloudImage.camera, 'camera_enu') and cloudImage.camera.camera_enu is not None:
        DrawAltAzGrid_v2(ax, cloudImage.GetAltAzGrid_fromcamera(buffer=True), **grid_kwargs)
    else:
        wcs_adapter = WCSCoordinateSystemsAdapter(cloudImage)
        DrawAltAzGrid_v2(ax, wcs_adapter.aazgrid, **grid_kwargs)

    # Set exact limits to match image dimensions
    ax.set_xlim(-0.5, w_px - 0.5)
    ax.set_ylim(h_px - 0.5, -0.5)

    # Save if filename provided
    if filename is not None:
        fig.savefig(filename, dpi=dpi, pad_inches=0)

    # Close if requested
    if close_figure:
        plt.close(fig)

    return fig, ax


def PlotAltAzGrid(cloudImage: CloudImage, outImageDir = None,  stars = False, showplot=True, from_camera = True, ax=None):
        if ax is None:
            doPlot=True
            fig, ax = plt.subplots(figsize=(20,10))
        else:
            doPlot=False
            fig=ax.figure
        ax.imshow(cloudImage.imagearray)
        if from_camera and cloudImage.camera.camera_enu is not None:
            DrawAltAzGrid(ax, cloudImage.GetAltAzGrid_fromcamera())
        else:
            from sudrabainiemakoni.wcs_coordinate_systems import WCSCoordinateSystemsAdapter
            wcs_adapter = WCSCoordinateSystemsAdapter(cloudImage)
            DrawAltAzGrid(ax, wcs_adapter.aazgrid)
        #ax.set_yticklabels([])
        #ax.set_xticklabels([])
        #ax.set_yticks([])
        #ax.set_xticks([])
        imid = cloudImage.code
        if stars:
            PlotStars(cloudImage, ax)
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}/horiz_coord_{imid}_zvaigznes.jpg', dpi=300, bbox_inches='tight')
        else:
            if outImageDir is not None:
                fig.savefig(f'{outImageDir}/horiz_coord_{imid}.jpg', dpi=300, bbox_inches='tight')
        if doPlot:
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
def PlotValidHeightPoints(imagearray, epilines, pts, heightpoints, valid, filename=None, ax=None):
    if ax is None:
        doPlot=True
        fig, ax = plt.subplots(figsize=(20,10))
    else:
        doPlot=False
        fig=ax.figure
        
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
    if ax is None:
        fig.colorbar(cs)
    
    
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    if doPlot:
        if filename is None:
            plt.show()
        else:
            plt.close()    


def InitPlotReferencedImages(webmerc: WebMercatorImage,
                             lonmin=15, lonmax=30, latmin=56, latmax=62):
    import tilemapbase
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()
    extent = tilemapbase.Extent.from_lonlat(lonmin,lonmax,latmin,latmax)
    e1=tilemapbase.Extent.from_3857(webmerc.xmin,webmerc.xmax, webmerc.ymax, webmerc.ymin)
    image_bounds = e1.to_project_web_mercator()
    plotter = tilemapbase.Plotter(extent, t, width=500)
    return image_bounds, plotter, t


def PlotReferencedImages(webmerc: WebMercatorImage,
                         projected_images,
                         camera_points=[],
                         outputFileName = None, showplot = False,
                         lonmin=15, lonmax=30, latmin=56, latmax=62,
                         alpha=0.8,
                         ax=None,
                         initData = None,
                         plotMap = True,
                         callback=None,):
    import tilemapbase
    if initData is None:
        tilemapbase.init(create=True)
        t = tilemapbase.tiles.build_OSM()
        extent = tilemapbase.Extent.from_lonlat(lonmin,lonmax,latmin,latmax)
        e1=tilemapbase.Extent.from_3857(webmerc.xmin,webmerc.xmax, webmerc.ymax, webmerc.ymin)
        image_bounds = e1.to_project_web_mercator()
        plotter = tilemapbase.Plotter(extent, t, width=500)
    else:
        image_bounds, plotter, t = initData
    import matplotlib.transforms
    w=16
    h=9*w/16
    hbb=7
    if ax is None:
        doPlot=True
        fig, ax = plt.subplots(figsize=(w,h), facecolor='#FAFAFA')
    else:
        doPlot=False
        fig=ax.figure
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if plotMap:
        plotter.plot(ax, t)
    try:
        alphas=list(alpha)
    except:
        alphas = [alpha]*len(projected_images)
    csl=[]
    for projected_image, _alpha in zip(projected_images, alphas):
        cs=ax.imshow(projected_image, extent=(image_bounds.xmin, image_bounds.xmax, image_bounds.ymax, image_bounds.ymin), alpha=_alpha)
        csl.append(cs)

    for plonlat in camera_points:
        p = tilemapbase.project(plonlat[0], plonlat[1])
        ax.plot(p[0],p[1],marker='o', ms=12)
    if callable(callback):
        callback(ax)
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=300, bbox_inches='tight')
    if doPlot:
        if showplot:
            plt.show()
        else:
            plt.close()
    else:
        return csl
