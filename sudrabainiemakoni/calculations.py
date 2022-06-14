import numpy as np
def GetImageRaDecGrid(imgarr, wcs):
    ii, jj = np.meshgrid(np.arange(0,imgarr.shape[1]),np.arange(0,imgarr.shape[0]))
    coordgrid = wcs.pixel_to_world(*[ii.flatten(),jj.flatten()])
    coordgrid=coordgrid.reshape(ii.shape)
    return coordgrid

