import pandas as pd
import numpy as np
import sys, os
from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni import plots
from PyQt5.QtWidgets import QApplication
from qthelper import gui_fname,gui_dir,gui_string
App = QApplication(sys.argv)

filename_jpg=None
if filename_jpg is None:
    filename_jpg=gui_fname(caption="Sudrabaino mākoņu attēls...", filter="(*.jpg)")
if filename_jpg=="":
    sys.exit()
filename_stars=os.path.splitext(filename_jpg)[0]+'_zvaigznes.txt'
if not os.path.exists(filename_stars):
    filename_stars=gui_fname(caption="Zvaigžņu fails...", filter="(*.txt)")
case_id=os.path.splitext(os.path.split(filename_jpg)[1])[0]
print(f'Id:{case_id}')
print(f'Fails:{filename_jpg}')
print(f'Zvaigznes:{filename_stars}')

latlonfile=os.path.splitext(filename_jpg)[0]+'_latlon.txt'
lat, lon = None, None

if os.path.exists(latlonfile):
    print(f'LatLon fails:{latlonfile}')
    try:
        df=pd.read_csv(latlonfile, sep=',', header=None)
        print(df)
        lat,lon=df.iloc[0,0],df.iloc[0,1]
    except:
        raise

if lat is  None or lon is  None:
    slatlon = gui_string(caption='Platums,Garums')
    try:
        sl=slatlon.split(',')
        lat, lon = float(sl[0]), float(sl[1])
    except:
        pass

resultdir=os.path.split(filename_jpg)[0]
resultdir=gui_dir(directory=resultdir, caption="Rezultātu katalogs")
resultproj=f'{resultdir}/{os.path.splitext(os.path.split(filename_jpg)[1])[0]}.proj'
cldim = CloudImage(case_id, filename_jpg)
cldim.setDateFromExif()
if lat is not None and lon is not None:
    cldim.setLocation(lat=lat, lon=lon)
else:
    cldim.setLocationExif()
print('UTC:', cldim.date)
print(cldim.location.to_geodetic())
# uzstādām zvaigžņu sarakstu
df = pd.read_csv(filename_stars, sep='\t', header=None)
# zvaigžņu nosaukumi pirmajā kolonā
starnames = df[0]
# atbilstošās pikseļu koordinātes otrajā un trešajā kolonā
pixels=np.array(df[[1,2]])
cldim.setStarReferences(starnames, pixels)
# izdrukājam zvaigžņu ekvatoriālās un pikseļu koordinātes pārbaudes nolūkos
print(cldim.getSkyCoords())
print(cldim.getPixelCoords())
cldim.PrepareCamera()
az, el, rot = cldim.camera.get_azimuth_elevation_rotation()
print(f'Kameras ass azimuts {az:.2f}°')
print(f'Kameras ass augstums virs horizonta {el:.2f}°')
print(f'Kameras pagrieziena leņķis {rot:.2f}°')
# noglabājam dotajam attēlam atbilstošo projektu - tas saturēs norādi uz attēlu, kameras, zvaigznes, novērotāja pozīciju, koordinātu sistēmas
# šo failu var vēlāk ielasīt ar CloudImage.load
cldim.save(resultproj)
print(f'Noglabāts:{resultproj}')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,10))
plots.PlotAltAzGrid(cldim,   stars = False, showplot=False, from_camera = True, ax=ax)
# zvaigžņu koordinātes enu sistēmā, vienības attālumam
enu_unit_coords = cldim.get_stars_enu_unit_coords()
# zvaigžņu pikseļu koordinātes atbilstoši referencētai kamerai
campx=cldim.camera.camera_enu.imageFromSpace(enu_unit_coords)
# ievadītās zvaigžņu pikseļu koordinātes
pxls = cldim.getPixelCoords()
for sr, cpx in zip(cldim.starReferences, campx):
    ix, iy = sr.pixelcoords
    p=ax.plot(ix,iy, marker='o', fillstyle='none')
    ax.annotate(sr.name, xy=(ix,iy), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
    ax.plot(cpx[0],cpx[1], marker='x', fillstyle='none', color=p[0].get_color())
#fig.savefig(f'{results_directory}/{case_id}_horizkoord.jpg', dpi=300, bbox_inches='tight')
plt.show()