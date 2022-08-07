import sys, os, shutil
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui, QtCore
from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni import plots
from smgui import Ui_MainWindow
from qthelper import gui_fname
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
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)
        
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
            # ievadītās zvaigžņu pikseļu koordinātes
            pxls = cldim.getPixelCoords()
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

            
if __name__ == '__main__':

    app = QApplication(sys.argv)
    myapp = MainW()
    myapp.show()
    sys.exit(app.exec_())
