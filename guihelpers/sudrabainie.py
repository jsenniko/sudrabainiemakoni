import sys, os, shutil
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui, QtCore
#from matplotlib.backend_tools import Cursors
from sudrabainiemakoni.cloudimage import CloudImage, StarReference
from sudrabainiemakoni import plots
from smgui import Ui_MainWindow
from qthelper import gui_fname, gui_string
import smhelper

class Stream(QtCore.QObject):
    #https://stackoverflow.com/a/44433766
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
    def isatty(self):
        #https://pythontechworld.com/issue/astropy/astropy/13351
        return False
    def flush(self):
        pass
#pyuic5 smgui.ui -o smgui.py
class MainW (QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.cloudimage = None
        self.setupUi(self)
        
        
        self.actionIelas_t_att_lu.triggered.connect(self.IelasitAtteluClick)
        self.actionKalibr_t_kameru.triggered.connect(self.KalibretKameruClick)
        self.actionSaglab_t_projektu.triggered.connect(self.SaglabatProjektu)
        self.actionIelas_t_projektu.triggered.connect(self.NolasitProjektu)
        self.actionHorizont_lo_koordin_tu_re_is.triggered.connect(self.ZimetAltAzClick)
        self.actionAtt_lu.triggered.connect(self.ZimetAttelu)
        self.actionCiparot_zvaigznes.triggered.connect(self.CiparotZvaigznesClick)
        
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)
        
        self.isCiparotZvaigznes = None
        
    def onUpdateText(self, text):
        #https://stackoverflow.com/a/44433766
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()
    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return standard output to defaults.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

    def IelasitAtteluClick(self):
        filename_jpg = gui_fname(caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg!='':
            self.IelasitAttelu(filename_jpg)
    def IelasitAttelu(self, filename_jpg):
        self.console.clear()
        case_id=os.path.splitext(os.path.split(filename_jpg)[1])[0]
        filename_stars = smhelper.check_stars_file(filename_jpg)
        lat,lon = smhelper.check_latlon_file(filename_jpg)
        self.cloudimage = CloudImage.from_files(case_id, filename_jpg, filename_stars, lat, lon)
        self.ZimetAttelu()
    def KalibretKameruClick(self):
        if self.cloudimage is not None:
            cldim = self.cloudimage
            cldim.PrepareCamera()
            az, el, rot = cldim.camera.get_azimuth_elevation_rotation()            
            print(f'Kameras ass azimuts {az:.2f}°')
            print(f'Kameras ass augstums virs horizonta {el:.2f}°')
            print(f'Kameras pagrieziena leņķis {rot:.2f}°')
            fx,fy = cldim.camera.get_focal_lengths_mm()
            print(f'Kameras fokusa attālumi (35mm ekvivalents) {fx:.1f} {fy:.1f}')
    def NolasitProjektu(self):
        projfile, _ = QFileDialog.getOpenFileName( 
                                               filter='(*.proj)',
                                               caption='Projekta fails')
        self.console.clear()
        self.cloudimage = CloudImage.load(projfile)
        print(f'Loaded project file {projfile}')
        print(self.cloudimage)
        self.ZimetAttelu()
        
    def SaglabatProjektu(self):
        projfile = os.path.splitext(self.cloudimage.filename)[0]+'.proj'
        projfile, _ = QFileDialog.getSaveFileName(directory=projfile, 
                                               filter='(*.proj)',
                                               caption='Projekta fails')
        if projfile!='':
            self.cloudimage.save(projfile)
            print(f'Saved project file {projfile}')
    def ZimetAltAzClick(self):       
        if self.cloudimage is not None:
            ax=self.MplWidget1.canvas.ax                        
            ax.clear()
            plots.PlotAltAzGrid(self.cloudimage, ax=ax)
            cldim = self.cloudimage
            # zvaigžņu koordinātes enu sistēmā, vienības attālumam
            enu_unit_coords = cldim.get_stars_enu_unit_coords()
            # zvaigžņu pikseļu koordinātes atbilstoši referencētai kamerai
            campx=cldim.camera.camera_enu.imageFromSpace(enu_unit_coords)            
            for sr, cpx in zip(cldim.starReferences, campx):
                ix, iy = sr.pixelcoords
                p=ax.plot(ix,iy, marker='o', fillstyle='none')
                ax.annotate(sr.name, xy=(ix,iy), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                ax.plot(cpx[0],cpx[1], marker='x', fillstyle='none', color=p[0].get_color())
            self.MplWidget1.canvas.draw()
    def ZimetAttelu(self):
        if self.cloudimage is not None:
            ax=self.MplWidget1.canvas.ax            
            ax.clear()
            #cid = fig.canvas.mpl_connect('button_press_event', onclick)
            ax.imshow(self.cloudimage.imagearray)
            plots.PlotStars(self.cloudimage, ax)
            
            self.MplWidget1.canvas.draw()
    def onclick_ciparotzvaigznes(self, event):
        #https://stackoverflow.com/a/64486726
        ax=event.inaxes
        #print(event)
        try: # use try/except in case we are not using Qt backend
            zooming_panning = ( ax.figure.canvas.cursor().shape() not in  [0,13] ) # 0 is the arrow, which means we are not zooming or panning.
        except:
            zooming_panning = False
        if zooming_panning:
            #print("Zooming or panning")
            return
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.button==1:
            X_coordinate = event.xdata
            Y_coordinate = event.ydata
            sname=gui_string(caption='Ievadi zvaigznes vārdu')
            if sname is not None:
                ax.plot(X_coordinate,Y_coordinate, marker='o', fillstyle='none')
                ax.annotate(sname, xy=(X_coordinate,Y_coordinate), xytext=(3,3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                #starlist.append((sname,X_coordinate,Y_coordinate))
                ax.figure.canvas.draw()
                
                cldim = self.cloudimage
                sr=StarReference(sname, [X_coordinate,Y_coordinate])
                Ok = False
                try:
                    sr.getSkyCoord()
                    Ok = True
                except Exception as e:                    
                    print(e)
                    Ok = gui_string(caption='Neatpazīst zvaigzni, vai ievadīt?') is not None
                if Ok:
                    cldim.starReferences.append(sr)
                
        else:
            self.StopCiparotZvaigznes()
        
    def StartCiparotZvaigznes(self):
        if self.isCiparotZvaigznes is None:
            self.MplWidget1.canvas.fig.set_facecolor('mistyrose')
            self.ZimetAttelu()
            self.isCiparotZvaigznes = self.MplWidget1.canvas.mpl_connect('button_press_event', self.onclick_ciparotzvaigznes)            
    def StopCiparotZvaigznes(self):
        if self.isCiparotZvaigznes is not None:
            self.MplWidget1.canvas.mpl_disconnect(self.isCiparotZvaigznes)
            filename_stars=os.path.splitext(self.cloudimage.filename)[0]+'_zvaigznes.txt'
            self.cloudimage.saveStarReferences(filename_stars)
            self.MplWidget1.canvas.fig.set_facecolor('white')
            self.ZimetAttelu()
        self.isCiparotZvaigznes = None
    def CiparotZvaigznesClick(self):
        if self.isCiparotZvaigznes is None:
            self.StartCiparotZvaigznes()
        else:
            self.StopCiparotZvaigznes()
        
            
if __name__ == '__main__':

    app = QApplication(sys.argv)
    myapp = MainW()
    myapp.show()
    sys.exit(app.exec_())
