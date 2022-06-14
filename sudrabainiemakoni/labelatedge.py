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