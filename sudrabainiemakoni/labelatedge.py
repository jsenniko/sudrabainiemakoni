#Adapted from https://numbersmithy.com/how-to-label-the-contour-lines-at-the-edge-of-a-matplotlib-plot/
import numpy as np
def labelAtEdge(levels, cs, ax, fmt=None, side='both', pad=0.005, eps=1e-5, **kwargs):
    '''Label contour lines at the edge of plot

    Args:
        levels (1d array): contour levels.
        cs (QuadContourSet obj): the return value of contour() function.
        ax (Axes obj): matplotlib axis.
        fmt lambda function retruning str
    Keyword Args:
        side (str): on which side of the plot intersections of contour lines
            and plot boundary are checked. Could be: 'left', 'right', 'top',
            'bottom' or 'all'. E.g. 'left' means only intersections of contour
            lines and left plot boundary will be labeled. 'all' means all 4
            edges.
        pad (float): padding to add between plot edge and label text.
        **kwargs: additional keyword arguments to control texts. E.g. fontsize,
            color.
    '''

    from matplotlib.transforms import Bbox
    collections = cs.collections
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = Bbox.from_bounds(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])

    if fmt is None:
        fmtfunc=lambda x:str(x)
    else:
        if isinstance(fmt, str):
            fmtfunc=lambda x: fmt % x
        else:
            fmtfunc=fmt
#eps = 1e-5  # error for checking boundary intersection

    # -----------Loop through contour levels-----------
    for ii, lii in enumerate(levels):

        cii = collections[ii]  # contours for level lii
        pathsii = cii.get_paths()  # the Paths for these contours
        if len(pathsii) == 0:
            continue

        for pjj in pathsii:

            # check first whether the contour intersects the axis boundary
            if not pjj.intersects_bbox(bbox, False):  # False significant here
                continue

            xjj = pjj.vertices[:, 0]
            yjj = pjj.vertices[:, 1]

            # intersection with the left edge
            if side in ['left', 'all']:
                inter_idx = np.where(abs(xjj-xlim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]

                    ax.text(inter_x-pad, inter_y, fmtfunc(lii),
                            ha='right',
                            va='center',
                            **kwargs)

            # intersection with the right edge
            if side in ['right', 'all']:
                inter_idx = np.where(abs(xjj-xlim[1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]

                    ax.text(inter_x+pad, inter_y, fmtfunc(lii),
                            ha='left',
                            va='center',
                            **kwargs)

            # intersection with the bottom edge
            if side in ['bottom', 'all']:

                inter_idx = np.where(abs(yjj-ylim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x, inter_y-pad, fmtfunc(lii),
                            ha='center',
                            va='top',
                            **kwargs)

            # intersection with the top edge
            if side in ['top', 'all']:
                inter_idx = np.where(abs(yjj-ylim[-1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]

                    ax.text(inter_x, inter_y+pad, fmtfunc(lii),
                            ha='center',
                            va='bottom',
                            **kwargs)

    return


def _compute_line_segment_intersection(p1, p2, boundary_pos, axis='x'):
    """
    Compute exact intersection of line segment with vertical or horizontal boundary.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        boundary_pos: Position of boundary (x value for vertical, y for horizontal)
        axis: 'x' for vertical boundary, 'y' for horizontal boundary

    Returns:
        Intersection point (x, y) or None if no intersection
    """
    x1, y1 = p1
    x2, y2 = p2

    if axis == 'x':
        if abs(x2 - x1) < 1e-10:
            if abs(x1 - boundary_pos) < 1e-6:
                return (boundary_pos, y1)
            return None

        t = (boundary_pos - x1) / (x2 - x1)
        if -1e-6 <= t <= 1 + 1e-6:
            y_intersect = y1 + t * (y2 - y1)
            return (boundary_pos, y_intersect)
    else:
        if abs(y2 - y1) < 1e-10:
            if abs(y1 - boundary_pos) < 1e-6:
                return (x1, boundary_pos)
            return None

        t = (boundary_pos - y1) / (y2 - y1)
        if -1e-6 <= t <= 1 + 1e-6:
            x_intersect = x1 + t * (x2 - x1)
            return (x_intersect, boundary_pos)

    return None


def labelAtEdge_v2(levels, cs, ax, fmt=None, side='both', pad=0.005,
                   label_levels=None, placement='outside', **kwargs):
    """
    Label contour lines at the edge of plot with geometric intersection calculation.

    Args:
        levels (1d array): contour levels.
        cs (QuadContourSet obj): the return value of contour() function.
        ax (Axes obj): matplotlib axis.
        fmt: Format string or lambda function returning str

    Keyword Args:
        side (str): on which side of the plot intersections of contour lines
            and plot boundary are checked. Could be: 'left', 'right', 'top',
            'bottom' or 'all'.
        pad (float): padding to add between plot edge and label text.
        label_levels (array-like): If provided, only label these specific levels.
            If None, all levels are labeled.
        placement (str): 'outside' - labels outside plot boundary (default)
                        'inside' - labels inside plot boundary
        **kwargs: additional keyword arguments to control texts. E.g. fontsize, color.
    """
    from matplotlib.transforms import Bbox
    collections = cs.collections
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = Bbox.from_bounds(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])

    if fmt is None:
        fmtfunc = lambda x: str(x)
    else:
        if isinstance(fmt, str):
            fmtfunc = lambda x: fmt % x
        else:
            fmtfunc = fmt

    if label_levels is not None:
        label_levels = set(label_levels)

    inside = (placement == 'inside')

    labeled_positions = {s: set() for s in ['left', 'right', 'top', 'bottom']}
    min_label_distance = 20

    y_min = min(ylim)
    y_max = max(ylim)
    x_min = min(xlim)
    x_max = max(xlim)

    for ii, lii in enumerate(levels):
        if label_levels is not None and lii not in label_levels:
            continue

        cii = collections[ii]
        pathsii = cii.get_paths()
        if len(pathsii) == 0:
            continue

        for pjj in pathsii:
            if not pjj.intersects_bbox(bbox, False):
                continue

            vertices = pjj.vertices
            intersections_left = []
            intersections_right = []
            intersections_top = []
            intersections_bottom = []

            for i in range(len(vertices) - 1):
                p1 = vertices[i]
                p2 = vertices[i + 1]

                if side in ['left', 'all']:
                    intersection = _compute_line_segment_intersection(p1, p2, xlim[0], 'x')
                    if intersection:
                        inter_x, inter_y = intersection
                        if y_min <= inter_y <= y_max:
                            intersections_left.append((inter_x, inter_y))

                if side in ['right', 'all']:
                    intersection = _compute_line_segment_intersection(p1, p2, xlim[1], 'x')
                    if intersection:
                        inter_x, inter_y = intersection
                        if y_min <= inter_y <= y_max:
                            intersections_right.append((inter_x, inter_y))

                if side in ['bottom', 'all']:
                    intersection = _compute_line_segment_intersection(p1, p2, ylim[0], 'y')
                    if intersection:
                        inter_x, inter_y = intersection
                        if x_min <= inter_x <= x_max:
                            intersections_bottom.append((inter_x, inter_y))

                if side in ['top', 'all']:
                    intersection = _compute_line_segment_intersection(p1, p2, ylim[1], 'y')
                    if intersection:
                        inter_x, inter_y = intersection
                        if x_min <= inter_x <= x_max:
                            intersections_top.append((inter_x, inter_y))

            def add_label_if_not_too_close_vertical(intersections, edge_name, text_kwargs):
                if len(intersections) > 0:
                    inter_x, inter_y = intersections[len(intersections) // 2]

                    too_close = False
                    for prev_y in labeled_positions[edge_name]:
                        if abs(inter_y - prev_y) < min_label_distance:
                            too_close = True
                            break

                    if not too_close:
                        ax.text(**text_kwargs)
                        labeled_positions[edge_name].add(inter_y)

            def add_label_if_not_too_close_horizontal(intersections, edge_name, text_kwargs):
                if len(intersections) > 0:
                    inter_x, inter_y = intersections[len(intersections) // 2]

                    too_close = False
                    for prev_x in labeled_positions[edge_name]:
                        if abs(inter_x - prev_x) < min_label_distance:
                            too_close = True
                            break

                    if not too_close:
                        ax.text(**text_kwargs)
                        labeled_positions[edge_name].add(inter_x)

            if intersections_left:
                inter_x, inter_y = intersections_left[0]
                if inside:
                    add_label_if_not_too_close_vertical(intersections_left, 'left',
                        {'x': inter_x + pad, 'y': inter_y, 's': fmtfunc(lii), 'ha': 'left', 'va': 'center', **kwargs})
                else:
                    add_label_if_not_too_close_vertical(intersections_left, 'left',
                        {'x': inter_x - pad, 'y': inter_y, 's': fmtfunc(lii), 'ha': 'right', 'va': 'center', **kwargs})

            if intersections_right:
                inter_x, inter_y = intersections_right[0]
                if inside:
                    add_label_if_not_too_close_vertical(intersections_right, 'right',
                        {'x': inter_x - pad, 'y': inter_y, 's': fmtfunc(lii), 'ha': 'right', 'va': 'center', **kwargs})
                else:
                    add_label_if_not_too_close_vertical(intersections_right, 'right',
                        {'x': inter_x + pad, 'y': inter_y, 's': fmtfunc(lii), 'ha': 'left', 'va': 'center', **kwargs})

            if intersections_bottom:
                inter_x, inter_y = intersections_bottom[0]
                if inside:
                    add_label_if_not_too_close_horizontal(intersections_bottom, 'bottom',
                        {'x': inter_x, 'y': inter_y - pad, 's': fmtfunc(lii), 'ha': 'center', 'va': 'bottom', **kwargs})
                else:
                    add_label_if_not_too_close_horizontal(intersections_bottom, 'bottom',
                        {'x': inter_x, 'y': inter_y + pad, 's': fmtfunc(lii), 'ha': 'center', 'va': 'top', **kwargs})

            if intersections_top:
                inter_x, inter_y = intersections_top[0]
                if inside:
                    add_label_if_not_too_close_horizontal(intersections_top, 'top',
                        {'x': inter_x, 'y': inter_y + pad, 's': fmtfunc(lii), 'ha': 'center', 'va': 'top', **kwargs})
                else:
                    add_label_if_not_too_close_horizontal(intersections_top, 'top',
                        {'x': inter_x, 'y': inter_y - pad, 's': fmtfunc(lii), 'ha': 'center', 'va': 'bottom', **kwargs})

    return