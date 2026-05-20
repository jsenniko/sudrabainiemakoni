__author__ = 'Juris Seņņikovs'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymap3d


def PlotAltAzGrid_v2(imagearray, camera, showplot=True,
                      ax=None, grid_kwargs=None, exact_size=False):
    """
    Plot altitude/azimuth grid with improved styling options.

    Args:
        stars: Whether to plot star references
        showplot: Whether to display the plot
        ax: Existing axes to plot on (if None, creates new figure)
        grid_kwargs: Dictionary of parameters to pass to DrawAltAzGrid_v2
        exact_size: If True, create figure matching exact image dimensions for overlay
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    if exact_size:
        h_px, w_px = imagearray.shape[:2]
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

    ax.imshow(imagearray)

    h_px, w_px = imagearray.shape[:2]
    DrawAltAzGrid_direct(ax, camera, **grid_kwargs)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])


    if doPlot:
        if showplot:
            plt.show()
        else:
            plt.close()


def CreateGridOverlayFigure(imagearray, camera, grid_kwargs=None, dpi=100,
                             filename=None, close_figure=False,
                             image_width=None, image_height=None):
    """
    Create a figure with exact image dimensions and grid overlay.

    Args:
        imagearray: numpy image array (H x W x 3), or None for a grid-only figure
        camera: cameratransform Camera object
        grid_kwargs: keyword arguments forwarded to DrawAltAzGrid_direct
        dpi: DPI for the figure (default 100)
        filename: if provided, save the figure to this file
        close_figure: if True, close the figure after saving (default False)
        image_width: pixel width when imagearray is None (required if imagearray is None)
        image_height: pixel height when imagearray is None (required if imagearray is None)

    Returns:
        Tuple of (fig, ax) - matplotlib figure and axes objects
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    if imagearray is not None:
        h_px, w_px = imagearray.shape[:2]
    else:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width and image_height are required when imagearray is None"
            )
        w_px, h_px = int(image_width), int(image_height)

    fig, ax = plt.subplots(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')

    if imagearray is not None:
        ax.imshow(imagearray, aspect='auto', interpolation='none')
    else:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    ax.set_xlim(-0.5, w_px - 0.5)
    ax.set_ylim(h_px - 0.5, -0.5)

    grid_kwargs = grid_kwargs.copy()
    grid_kwargs['label_placement'] = 'inside'

    DrawAltAzGrid_direct(ax, camera, **grid_kwargs)

    if filename is not None:
        fig.savefig(filename, dpi=dpi, pad_inches=0)

    if close_figure:
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# Direct-projection grid (no contour artifacts near wrap-around / zenith)
# ---------------------------------------------------------------------------

def _az_line_enu(az_deg, alt_steps):
    """Return ENU unit vectors along one azimuth meridian for a range of altitudes."""
    az_rad  = np.radians(az_deg)
    alt_rad = np.radians(alt_steps)
    e =  np.cos(alt_rad) * np.sin(az_rad)
    n =  np.cos(alt_rad) * np.cos(az_rad)
    u =  np.sin(alt_rad)
    return np.column_stack([e, n, u])


def _alt_line_enu(alt_deg, az_steps):
    """Return ENU unit vectors along one altitude parallel for a range of azimuths."""
    alt_rad = np.radians(alt_deg)
    az_rad  = np.radians(az_steps)
    e =  np.cos(alt_rad) * np.sin(az_rad)
    n =  np.cos(alt_rad) * np.cos(az_rad)
    u =  np.full(len(az_steps), np.sin(alt_rad))
    return np.column_stack([e, n, u])


def _project_and_split(camera, enu_points, x_min, x_max, y_min, y_max):
    """
    Project ENU points through camera, return list of (xs, ys) segments
    split at points that fall outside the image or where the projected line
    jumps discontinuously (e.g. goes behind the camera).
    """
    px = camera.imageFromSpace(enu_points, hide_backpoints=True)
    # hide_backpoints sets out-of-hemisphere points to NaN
    xs, ys = px[:, 0], px[:, 1]

    # Also mask points outside image bounds with a margin
    margin = 5
    outside = (xs < x_min - margin) | (xs > x_max + margin) | \
              (ys < y_min - margin) | (ys > y_max + margin)
    xs = np.where(outside, np.nan, xs)
    ys = np.where(outside, np.nan, ys)

    # Split into contiguous segments at NaN boundaries
    segments = []
    seg_x, seg_y = [], []
    for x, y in zip(xs, ys):
        if np.isnan(x) or np.isnan(y):
            if len(seg_x) > 1:
                segments.append((np.array(seg_x), np.array(seg_y)))
            seg_x, seg_y = [], []
        else:
            seg_x.append(x)
            seg_y.append(y)
    if len(seg_x) > 1:
        segments.append((np.array(seg_x), np.array(seg_y)))
    return segments


def DrawAltAzGrid_direct(ax, camera,
                         azlevels=None, altlevels=None,
                         nticks=15, nticks_major=None,
                         minor_az_step=10, major_az_step=30,
                         minor_alt_step=10, major_alt_step=30,
                         alt_range=(-10, 90), alt_sample_step=0.2,
                         az_sample_step=0.2,
                         minor_az_style=None, major_az_style=None,
                         minor_alt_style=None, major_alt_style=None,
                         label_pad=10, label_kwargs=None,
                         label_fmt=None,
                         label_major_only=True,
                         label_placement='outside',
                         label_side='all',
                         min_label_sep=30):
    """
    Draw az/alt grid by projecting lines directly through the camera model.

    Avoids contour() wrap-around artifacts near the 0/360 boundary and zenith.

    Parameters
    ----------
    ax : matplotlib Axes  (must already contain the image so xlim/ylim are set)
    camera : cameratransform Camera
    azlevels : array-like, optional — explicit azimuth levels; overrides step-based generation
    altlevels : array-like, optional — explicit altitude levels; overrides step-based generation
    nticks : int — max minor gridlines when steps are None (auto mode)
    nticks_major : int — max major gridlines when steps are None; defaults to nticks//3
    minor_az_step, major_az_step : float — azimuth grid spacing in degrees (manual mode)
    minor_alt_step, major_alt_step : float — altitude grid spacing in degrees (manual mode)
    alt_range : (min, max) altitude in degrees used when building step-based levels; default (-10, 90)
    alt_sample_step : float — sampling resolution along azimuth lines (degrees)
    az_sample_step : float — sampling resolution along altitude circles (degrees)
    minor_*_style, major_*_style : dict — line style kwargs passed to ax.plot()
    label_pad : float — pixel offset from edge for labels
    label_kwargs : dict — text kwargs for labels
    label_fmt : str, callable, or dict with keys 'az' and/or 'alt' — label format;
                str is used as % format, callable receives the level value,
                dict lets you specify different formats per coordinate
    label_major_only : bool — only label major gridlines
    label_placement : 'outside' or 'inside' — label offset direction relative to image edge
    label_side : 'all', 'left', 'right', 'top', 'bottom' — which edges to label
                 'all' maps az lines to top+bottom, alt lines to left+right
    min_label_sep : float — minimum pixel separation between labels on the same edge
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_min, x_max = min(xlim), max(xlim)
    y_min, y_max = min(ylim), max(ylim)

    if minor_az_style is None:
        minor_az_style  = dict(color='#DDFFDD', linestyle=':', linewidth=0.3)
    if major_az_style is None:
        major_az_style  = dict(color='#DDFFDD', linestyle='--', linewidth=0.8)
    if minor_alt_style is None:
        minor_alt_style = dict(color='#DDFFDD', linestyle=':', linewidth=0.3)
    if major_alt_style is None:
        major_alt_style = dict(color='#DDFFDD', linestyle='--', linewidth=0.8)
    if label_kwargs is None:
        label_kwargs = dict(color='white', fontsize=7, fontweight='bold')

    # Resolve label format functions
    default_az_fmt  = lambda v: f'{v % 360:.0f}°'
    default_alt_fmt = lambda v: f'{v:.0f}°'
    if label_fmt is None:
        az_fmt  = default_az_fmt
        alt_fmt = default_alt_fmt
    elif isinstance(label_fmt, dict):
        raw_az  = label_fmt.get('az',  default_az_fmt)
        raw_alt = label_fmt.get('alt', default_alt_fmt)
        az_fmt  = (lambda fmt: lambda v: fmt % v)(raw_az)  if isinstance(raw_az,  str) else raw_az
        alt_fmt = (lambda fmt: lambda v: fmt % v)(raw_alt) if isinstance(raw_alt, str) else raw_alt
    elif isinstance(label_fmt, str):
        az_fmt  = lambda v, f=label_fmt: f % v
        alt_fmt = lambda v, f=label_fmt: f % v
    else:
        az_fmt  = label_fmt
        alt_fmt = label_fmt

    alt_steps = np.arange(alt_range[0], alt_range[1] + alt_sample_step, alt_sample_step)
    az_steps  = np.arange(0, 360 + az_sample_step, az_sample_step)

    # Build sets of major/minor levels
    if azlevels is not None:
        az_major = set(np.asarray(azlevels, dtype=float))
        az_minor = set()
    else:
        if minor_az_step is not None:
            az_minor_arr = np.arange(0, 360, minor_az_step)
        else:
            locator = matplotlib.ticker.MaxNLocator(nticks, steps=[1, 2, 5, 10], prune='both')
            az_minor_arr = locator.tick_values(0, 360)
        if major_az_step is not None:
            az_major_arr = np.arange(0, 360, major_az_step)
        else:
            n_maj = nticks_major if nticks_major is not None else max(3, nticks // 3)
            locator_maj = matplotlib.ticker.MaxNLocator(n_maj, steps=[1, 2, 5, 10], prune='both')
            az_major_arr = locator_maj.tick_values(0, 360)
        az_major = set(az_major_arr)
        az_minor = set(az_minor_arr) - az_major

    if altlevels is not None:
        alt_major = set(np.asarray(altlevels, dtype=float))
        alt_minor = set()
    else:
        alt_min_v, alt_max_v = alt_range
        if minor_alt_step is not None:
            alt_minor_arr = np.arange(alt_min_v, alt_max_v + 1, minor_alt_step)
        else:
            locator = matplotlib.ticker.MaxNLocator(nticks, steps=[1, 2, 5, 10], prune='both')
            alt_minor_arr = locator.tick_values(alt_min_v, alt_max_v)
        if major_alt_step is not None:
            alt_major_arr = np.arange(alt_min_v, alt_max_v + 1, major_alt_step)
        else:
            n_maj = nticks_major if nticks_major is not None else max(3, nticks // 3)
            locator_maj = matplotlib.ticker.MaxNLocator(n_maj, steps=[1, 2, 5, 10], prune='both')
            alt_major_arr = locator_maj.tick_values(alt_min_v, alt_max_v)
        alt_major = set(alt_major_arr)
        alt_minor = set(alt_minor_arr) - alt_major

    # Resolve which sides each line type uses for labels
    # az lines run top-to-bottom so they cross top/bottom edges
    # alt lines run left-to-right so they cross left/right edges
    if label_side == 'all':
        az_label_sides  = {'top', 'bottom'}
        alt_label_sides = {'left', 'right'}
    elif label_side in ('left', 'right'):
        az_label_sides  = set()
        alt_label_sides = {label_side}
    elif label_side in ('top', 'bottom'):
        az_label_sides  = {label_side}
        alt_label_sides = set()
    else:
        az_label_sides  = {'top', 'bottom'}
        alt_label_sides = {'left', 'right'}

    # Tracks placed label positions per edge side to prevent overlap.
    # Horizontal sides (top/bottom): store x coords. Vertical sides (left/right): store y coords.
    used_positions = {'top': [], 'bottom': [], 'left': [], 'right': []}

    def draw_lines(levels, enu_func, steps, style, do_label, allowed_sides, fmt_func):
        for level in sorted(levels):
            enu = enu_func(level, steps)
            segments = _project_and_split(camera, enu, x_min, x_max, y_min, y_max)
            for sx, sy in segments:
                ax.plot(sx, sy, **style)
            if do_label and segments and allowed_sides:
                _label_edge_direct(ax, segments, level, x_min, x_max, y_min, y_max,
                                   label_pad, label_kwargs, allowed_sides, label_placement,
                                   used_positions, min_label_sep, fmt_func)

    draw_lines(az_minor,  _az_line_enu,  alt_steps, minor_az_style,  not label_major_only, az_label_sides,  az_fmt)
    draw_lines(az_major,  _az_line_enu,  alt_steps, major_az_style,  True,                 az_label_sides,  az_fmt)
    draw_lines(alt_minor, _alt_line_enu, az_steps,  minor_alt_style, not label_major_only, alt_label_sides, alt_fmt)
    draw_lines(alt_major, _alt_line_enu, az_steps,  major_alt_style, True,                 alt_label_sides, alt_fmt)


def _label_edge_direct(ax, segments, level, x_min, x_max, y_min, y_max,
                       pad, label_kwargs, allowed_sides, placement,
                       used_positions, min_label_sep, fmt_func):
    """
    Place labels for a projected grid line at every allowed edge it crosses.

    For each allowed side, finds the nearest segment endpoint within threshold and
    draws a label there — so a line crossing both top and bottom gets two labels.
    Skips placement if it would overlap an already-drawn label on that side.

    Parameters
    ----------
    allowed_sides : set of strings from {'left', 'right', 'top', 'bottom'}
    placement : 'outside' — text beyond the edge; 'inside' — text inward from the edge
    used_positions : dict side -> list of already-used coordinates (mutated in place)
    min_label_sep : minimum pixel separation between labels on the same edge
    fmt_func : callable(level) -> str
    """
    fmt = fmt_func(level)

    edge_threshold = max(pad * 4, 30)

    side_dist = {
        'left':   lambda x, _: abs(x - x_min),
        'right':  lambda x, _: abs(x - x_max),
        'top':    lambda _, y: abs(y - y_min),
        'bottom': lambda _, y: abs(y - y_max),
    }
    side_coord = {
        'left':   lambda _, y: y,
        'right':  lambda _, y: y,
        'top':    lambda x, _: x,
        'bottom': lambda x, _: x,
    }

    all_candidates = []
    for sx, sy in segments:
        for x, y in [(sx[0], sy[0]), (sx[-1], sy[-1])]:
            for side in allowed_sides:
                d = side_dist[side](x, y)
                all_candidates.append((d, side, x, y))

    if not all_candidates:
        return

    inside = (placement == 'inside')
    drawn_any = False

    for side in sorted(allowed_sides):
        side_candidates = [(d, x, y) for d, s, x, y in all_candidates if s == side]
        if not side_candidates:
            continue
        d, x, y = min(side_candidates, key=lambda c: c[0])
        if d > edge_threshold:
            continue
        coord = side_coord[side](x, y)
        if any(abs(coord - prev) < min_label_sep for prev in used_positions[side]):
            continue
        used_positions[side].append(coord)
        drawn_any = True
        if side == 'left':
            tx = x + pad if inside else x - pad
            ax.text(tx, y, fmt, ha='left' if inside else 'right', va='center', **label_kwargs)
        elif side == 'right':
            tx = x - pad if inside else x + pad
            ax.text(tx, y, fmt, ha='right' if inside else 'left', va='center', **label_kwargs)
        elif side == 'top':
            ty = y + pad if inside else y - pad
            ax.text(x, ty, fmt, ha='center', va='top' if inside else 'bottom', **label_kwargs)
        else:
            ty = y - pad if inside else y + pad
            ax.text(x, ty, fmt, ha='center', va='bottom' if inside else 'top', **label_kwargs)

    if not drawn_any:
        d, side, x, y = min(all_candidates, key=lambda c: c[0])
        coord = side_coord[side](x, y)
        if any(abs(coord - prev) < min_label_sep for prev in used_positions[side]):
            return
        used_positions[side].append(coord)
        if side == 'left':
            tx = x + pad if inside else x - pad
            ax.text(tx, y, fmt, ha='left' if inside else 'right', va='center', **label_kwargs)
        elif side == 'right':
            tx = x - pad if inside else x + pad
            ax.text(tx, y, fmt, ha='right' if inside else 'left', va='center', **label_kwargs)
        elif side == 'top':
            ty = y + pad if inside else y - pad
            ax.text(x, ty, fmt, ha='center', va='top' if inside else 'bottom', **label_kwargs)
        else:
            ty = y - pad if inside else y + pad
            ax.text(x, ty, fmt, ha='center', va='bottom' if inside else 'top', **label_kwargs)
        if side == 'left':
            tx = x + pad if inside else x - pad
            ax.text(tx, y, fmt, ha='left' if inside else 'right', va='center', **label_kwargs)
        elif side == 'right':
            tx = x - pad if inside else x + pad
            ax.text(tx, y, fmt, ha='right' if inside else 'left', va='center', **label_kwargs)
        elif side == 'top':
            ty = y + pad if inside else y - pad
            ax.text(x, ty, fmt, ha='center', va='top' if inside else 'bottom', **label_kwargs)
        else:
            ty = y - pad if inside else y + pad
            ax.text(x, ty, fmt, ha='center', va='bottom' if inside else 'top', **label_kwargs)
