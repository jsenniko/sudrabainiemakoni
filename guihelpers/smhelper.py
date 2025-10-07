# -*- coding: utf-8 -*-
import os, shutil
from sudrabainiemakoni import utils

# Support both direct script execution and package import
try:
    from .qthelper import gui_fname, gui_string
except ImportError:
    from qthelper import gui_fname, gui_string
def check_latlon_file(filename_jpg):
    # garuma, platuma fails
    latlonfile=os.path.splitext(filename_jpg)[0]+'_latlon.txt'
    lat, lon, height = None, None, 0.0
    if os.path.exists(latlonfile):
        try:
            with open(latlonfile,'r') as f:
                s=f.readline()
                s=s.split(',')   
                lat,lon=float(s[0]), float(s[1])
                if len(s)>2:
                    height=float(s[2])
                print(f'LatLon file:{latlonfile}')
        except:
            pass
    if lat is  None or lon is  None:
        lat, lon = utils.getExifLatLon(filename_jpg)
        if lat is  None or lon is  None:
            slatlon = gui_string(caption='Platums,Garums')
            if slatlon is not None:
                try:
                    sl=slatlon.split(',')
                    lat, lon = float(sl[0]), float(sl[1])
                except:
                    pass
        print('AAA',lat,lon)
        if lat is not None and lon is not None:
            with open(latlonfile,'w') as f:
                s=f'{lat},{lon}'
                f.write(s)
    return lat, lon, height
def check_stars_file(filename_jpg):
    # zvaigžņu fails
    filename_stars=os.path.splitext(filename_jpg)[0]+'_zvaigznes.txt'
    if not os.path.exists(filename_stars):
        filename_stars_entered=gui_fname(caption="Zvaigžņu fails...", filter="(*.txt)")
        if filename_stars_entered!='':
            shutil.copyfile(filename_stars_entered, filename_stars)
        else:
            with open(filename_stars, "w"):
                pass
    return filename_stars