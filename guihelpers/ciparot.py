import sys, os, pandas as pd
from PyQt5.QtWidgets import QApplication
from qthelper import gui_fname,gui_dir,gui_string
App = QApplication(sys.argv)

import matplotlib.pyplot as plt
starlist=[]
def onclick(event):
    #https://stackoverflow.com/a/64486726
    try: # use try/except in case we are not using Qt backend
        zooming_panning = ( fig.canvas.cursor().shape() != 0 ) # 0 is the arrow, which means we are not zooming or panning.
    except:
        zooming_panning = False
    if zooming_panning:
        #print("Zooming or panning")
        return
    print(event)
    #print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    sname=gui_string(caption='Ievadi zvaigznes vārdu')
    if sname is not None:
        ax=event.inaxes
        ax.plot(X_coordinate,Y_coordinate, marker='o', fillstyle='none')
        ax.annotate(sname, xy=(X_coordinate,Y_coordinate), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
        starlist.append((sname,X_coordinate,Y_coordinate))
        ax.figure.canvas.draw()

filename_jpg=gui_fname(caption="Sudrabaino mākoņu attēls...", filter="(*.jpg)")
if filename_jpg=="":
    sys.exit()
filename_stars=os.path.splitext(filename_jpg)[0]+'_zvaigznes.txt'
if os.path.exists(filename_stars):
    df = pd.read_csv(filename_stars, sep='\t', header=None)
    starlist=[(r.iloc[0],r.iloc[1],r.iloc[2]) for i, r in df.iterrows()]
import skimage.io
img=skimage.io.imread(filename_jpg)
fig, ax = plt.subplots(figsize=(15,10))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax.imshow(img)
for sname, X_coordinate,Y_coordinate in starlist:
    ax.plot(X_coordinate,Y_coordinate, marker='o', fillstyle='none')
    ax.annotate(sname, xy=(X_coordinate,Y_coordinate), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
plt.show()
print(starlist)
resultdir=os.path.split(filename_jpg)[0]
resultdir=gui_fname(directory=resultdir, caption="Zvaigžnu fails")
if resultdir is not None:
    filename_stars=f'{resultdir}/{os.path.splitext(os.path.split(filename_jpg)[1])[0]}_zvaigznes.txt'
    df=pd.DataFrame(starlist, )
    df.to_csv(filename_stars,sep='\t',header=None, index=False)
