import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui, QtCore
#from matplotlib.backend_tools import Cursors
import tilemapbase
from sudrabainiemakoni.cloudimage import CloudImage, StarReference, WebMercatorImage, CloudImagePair, HeightMap
from sudrabainiemakoni import plots, utils
from smgui import Ui_MainWindow
from qthelper import gui_fname, gui_string
import smhelper


class Stream(QtCore.QObject):
    # https://stackoverflow.com/a/44433766
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def isatty(self):
        # https://pythontechworld.com/issue/astropy/astropy/13351
        return False

    def flush(self):
        pass
# pyuic5 smgui.ui -o smgui.py


class MainW (QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.showMaximized()

        self.actionIelas_t_att_lu.triggered.connect(self.IelasitAtteluClick)
        self.actionKalibr_t_kameru.triggered.connect(self.KalibretKameruClick)
        self.actionSaglab_t_projektu.triggered.connect(self.SaglabatProjektu)
        self.actionIelas_t_projektu.triggered.connect(self.NolasitProjektu)
        self.actionIelas_t_otro_projektu.triggered.connect(
            self.NolasitProjektu2)
        self.actionHorizont_lo_koordin_tu_re_is.triggered.connect(
            self.ZimetAltAzClick)
        self.actionAtt_lu.triggered.connect(self.ZimetAtteluClick)
        self.actionCiparot_zvaigznes.triggered.connect(
            self.CiparotZvaigznesClick)
        self.actionProjic_t.triggered.connect(
            lambda: self.ProjicetClick(tips=0))
        self.actionProjic_t_kop.triggered.connect(
            lambda: self.ProjicetClick(tips=2))
        self.actionProjic_t_no_augstumu_kartes.triggered.connect(
            lambda: self.ProjicetNoKartesClick(tips=0))
        self.actionProjic_t_kop_no_augstumu_kartes.triggered.connect(
            lambda: self.ProjicetNoKartesClick(tips=2))
        self.actionProjekcijas_apgabals.triggered.connect(self.MainitApgabalu)
        self.actionKartes_apgabals.triggered.connect(self.KartesApgabals)
        self.actionKontrolpunkti.triggered.connect(
            self.CiparotKontrolpunktusClick)
        self.actionIelas_t_kontrolpunktus.triggered.connect(
            self.IelasitKontrolpunktus)
        self.actionKontrolpunktu_augstumus.triggered.connect(
            self.ZimetKontrolpunktuAugstumus)
        self.actionIzveidot_augstumu_karti.triggered.connect(
            self.IzveidotAugstumuKarti)
        self.actionIelas_t_augstumu_karti.triggered.connect(
            self.IelasitAugstumuKarti)
        self.actionSaglab_t_augstumu_karti.triggered.connect(
            self.SaglabatAugstumuKarti)
        self.actionAugstumu_karti.triggered.connect(self.ZimetAugstumuKarti)
        self.actionKameras_kalibr_cijas_parametri.triggered.connect(
            self.KamerasKalibracijasParametri)
        self.actionSaglab_t_projic_to_att_lu_JPG.triggered.connect(
            lambda: self.SaglabatProjicetoAttelu(jpg=True))
        self.actionSaglab_t_projic_to_att_lu_TIFF.triggered.connect(
            lambda: self.SaglabatProjicetoAttelu(jpg=False))
        self.actionIelas_t_kameru.triggered.connect(self.IelasitKameru)
        self.actionSaglab_t_kameru.triggered.connect(self.SaglabatKameru)
        self.actionUzst_d_t_datumu.triggered.connect(self.UzstaditDatumu)
        self.actionUzst_d_t_platumu_garumu_augstumu.triggered.connect(self.UzstaditKoordinates)

        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        self.cloudimage = None
        self.cloudimage2 = None
        self.cpair = None
        self.webmerc = WebMercatorImage(self.cloudimage, 17, 33, 56, 63, 1.0)
        self.projHeight = 80  # km
        self.map_bounds = [17, 33, 56, 63]
        self.map_alpha = 0.85
        self.heightmap = None
        self.projected_image = None
        self.camera_calib_params = dict(
            distortion=False, centers=True, separate_x_y=True, projectiontype='rectilinear')

        self.isCiparotZvaigznes = None
        self.isCiparotKontrolpunkti = None
        self.meerit_events = None

    def pelekot(self):
        self.actionKalibr_t_kameru.setEnabled(self.cloudimage is not None)
        self.actionSaglab_t_projektu.setEnabled(self.cloudimage is not None)
        self.actionIelas_t_otro_projektu.setEnabled(
            self.cloudimage is not None)
        self.actionHorizont_lo_koordin_tu_re_is.setEnabled(
            self.cloudimage is not None and hasattr(self.cloudimage, 'camera'))
        self.actionAtt_lu.setEnabled(self.cloudimage is not None)
        self.actionCiparot_zvaigznes.setEnabled(self.cloudimage is not None)
        self.actionProjic_t.setEnabled(
            self.cloudimage is not None and hasattr(self.cloudimage, 'camera'))
        self.actionProjic_t_kop.setEnabled(self.cloudimage is not None and hasattr(
            self.cloudimage, 'camera') and self.cloudimage2 is not None)
        self.actionKontrolpunkti.setEnabled(
            self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionIelas_t_kontrolpunktus.setEnabled(
            self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionKontrolpunktu_augstumus.setEnabled(self.cpair is not None)
        self.actionIzveidot_augstumu_karti.setEnabled(self.cpair is not None)
        self.actionIelas_t_augstumu_karti.setEnabled(
            self.cloudimage is not None)
        self.actionSaglab_t_augstumu_karti.setEnabled(
            self.heightmap is not None)
        self.actionAugstumu_karti.setEnabled(self.heightmap is not None)
        self.actionProjic_t_no_augstumu_kartes.setEnabled(
            self.heightmap is not None and self.cloudimage is not None)
        self.actionProjic_t_kop_no_augstumu_kartes.setEnabled(
            self.heightmap is not None and self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionSaglab_t_projic_to_att_lu_JPG.setEnabled(
            self.projected_image is not None)
        self.actionSaglab_t_projic_to_att_lu_TIFF.setEnabled(
            self.projected_image is not None)
        self.actionIelas_t_kameru.setEnabled(self.cloudimage is not None)
        self.actionSaglab_t_kameru.setEnabled(self.cloudimage is not None and hasattr(self.cloudimage, 'camera'))
        self.actionUzst_d_t_datumu.setEnabled(self.cloudimage is not None)
        
    def onUpdateText(self, text):
        # https://stackoverflow.com/a/44433766
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
        filename_jpg = gui_fname(
            caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg != '':
            self.IelasitAttelu(filename_jpg)

    def IelasitAttelu(self, filename_jpg):
        self.cloudimage2 = None
        self.console.clear()
        case_id = os.path.splitext(os.path.split(filename_jpg)[1])[0]
        filename_stars = smhelper.check_stars_file(filename_jpg)
        lat, lon, height = smhelper.check_latlon_file(filename_jpg)
        self.cloudimage = CloudImage.from_files(
            case_id, filename_jpg, filename_stars, lat, lon, height=height)
        self.ZimetAttelu()
        self.pelekot()
    def UzstaditDatumu(self):
        try:
            d=self.cloudimage.date.to_datetime(timezone=CloudImage.timezone)
            s=d.strftime('%Y-%m-%dT%H:%M:%S')
        except:
            s=''
        s=gui_string(text=s, caption='Ievadi datumu, YYYY-MM-DDTHH:MM:SS')
        try:
            import datetime
            d=datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            self.cloudimage.setDate(d)
        except:
            print('Nepareizs datums!')
    def UzstaditKoordinates(self):
        try:
            lat,lon,height=self.cloudimage.getLocation()
            s=f'{lat:.5f},{lon:.5f},{height:.0f}'
            
        except:
            s=''
        try:
            s=gui_string(text=s, caption='Ievadi koordinātes (lat,lon,z)')
            s=s.split(',')   
            lat,lon,height=float(s[0]), float(s[1]),0.0
            if len(s)>2:
                height=float(s[2])
            self.cloudimage.setLocation(lon=lon, lat=lat, height=height)
        except:
            print('Nepareiza ievade!')
    def KalibretKameruClick(self):
        if self.cloudimage is not None:
            cldim = self.cloudimage
            cldim.PrepareCamera(**self.camera_calib_params)
            # distortion=False, centers=True, separate_x_y=True
            #cldim.PrepareCamera(method='optnew', distortion=args.optimizeDistortion, centers=args.notOptimizeCenter, separate_x_y=args.notOptimizeUnsymmetric)
            az, el, rot = cldim.camera.get_azimuth_elevation_rotation()
            print(f'Kameras ass azimuts {az:.2f}°')
            print(f'Kameras ass augstums virs horizonta {el:.2f}°')
            print(f'Kameras pagrieziena leņķis {rot:.2f}°')
            fx, fy, cx, cy = cldim.camera.get_focal_lengths_mm()
            print(
                f'Kameras fokusa attālumi (35mm ekvivalents) {fx:.1f} {fy:.1f}')
            print(f'Kameras ass pozīcija {cx:.1f} {cy:.1f}')
            self.ZimetAltAzClick()
        self.pelekot()

    def NolasitProjektu(self):

        projfile, _ = QFileDialog.getOpenFileName(
            filter='(*.proj)',
            caption='Projekta fails')
        if projfile != '':
            self.cloudimage2 = None
            self.cpair = None
            self.console.clear()
            self.cloudimage = CloudImage.load(projfile)
            print(f'Loaded project file {projfile}')
            print(self.cloudimage)
            self.ZimetAttelu()
        self.pelekot()

    def NolasitProjektu2(self):
        projfile, _ = QFileDialog.getOpenFileName(
            filter='(*.proj)',
            caption='Projekta fails')
        if projfile != '':
            self.cpair = None
            self.cloudimage2 = CloudImage.load(projfile)
            print(f'Loaded project file {projfile}')
            print(self.cloudimage2)
            self.ZimetAttelu(otrs=True)
        self.pelekot()

    def SaglabatProjektu(self):
        projfile = os.path.splitext(self.cloudimage.filename)[0]+'.proj'
        projfile, _ = QFileDialog.getSaveFileName(directory=projfile,
                                                  filter='(*.proj)',
                                                  caption='Projekta fails')
        if projfile != '':
            self.cloudimage.save(projfile)
            print(f'Saved project file {projfile}')
    def IelasitKameru(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0]+'_enu.json'
        camfile, _ = QFileDialog.getOpenFileName(
            directory=camfile,
            filter='(*.json)',
            caption='Kameras fails')
        if camfile !='':
            self.cloudimage.LoadCamera(camfile)
    def SaglabatKameru(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0]+'_enu.json'
        camfile, _ = QFileDialog.getSaveFileName(
            directory=camfile,
            filter='(*.json)',
            caption='Kameras fails')
        if camfile !='':
            self.cloudimage.SaveCamera(camfile)
        
    def ZimetAltAzClick(self):
        self.disconnect_meerit()
        if self.cloudimage is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            plots.PlotAltAzGrid(self.cloudimage, ax=ax)
            cldim = self.cloudimage
            # zvaigžņu koordinātes enu sistēmā, vienības attālumam
            enu_unit_coords = cldim.get_stars_enu_unit_coords()
            # zvaigžņu pikseļu koordinātes atbilstoši referencētai kamerai
            campx = cldim.camera.camera_enu.imageFromSpace(enu_unit_coords)
            for sr, cpx in zip(cldim.starReferences, campx):
                ix, iy = sr.pixelcoords
                p = ax.plot(ix, iy, marker='o', fillstyle='none')
                ax.annotate(sr.name, xy=(ix, iy), xytext=(
                    3, 3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                ax.plot(cpx[0], cpx[1], marker='x',
                        fillstyle='none', color=p[0].get_color())
            self.MplWidget1.canvas.draw()

    def ZimetAtteluClick(self):
        self.ZimetAttelu(otrs=self.cloudimage2 is not None)

    def ZimetAttelu(self, otrs=False, kontrolpunkti=False):
        self.disconnect_meerit()
        if self.cloudimage is not None:
            if otrs and self.cloudimage2 is not None:
                self.MplWidget1.canvas.initplot([121, 122])
                ax = self.MplWidget1.canvas.ax[0]
                ax2 = self.MplWidget1.canvas.ax[1]
            else:
                self.MplWidget1.canvas.initplot()
                ax = self.MplWidget1.canvas.ax
            #cid = fig.canvas.mpl_connect('button_press_event', onclick)
            ax.imshow(self.cloudimage.imagearray)
            if kontrolpunkti:
                self.plotMatches(ax, 0)
            else:
                plots.PlotStars(self.cloudimage, ax)
            if otrs and self.cloudimage2 is not None:
                ax2.imshow(self.cloudimage2.imagearray)
                if kontrolpunkti:
                    self.plotMatches(ax2, 1)
                else:
                    plots.PlotStars(self.cloudimage2, ax2)

            self.MplWidget1.canvas.draw()
            
    def onclick_ciparotzvaigznes(self, event):
        # https://stackoverflow.com/a/64486726
        ax = event.inaxes
        if ax is None:
            return
        # print(event)
        try:  # use try/except in case we are not using Qt backend
            # 0 is the arrow, which means we are not zooming or panning.
            zooming_panning = (
                ax.figure.canvas.cursor().shape() not in [0, 13])
        except:
            zooming_panning = False
        if zooming_panning:
            #print("Zooming or panning")
            return
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.button == 1:
            X_coordinate = event.xdata
            Y_coordinate = event.ydata
            sname = gui_string(caption='Ievadi zvaigznes vārdu')
            if sname is not None:
                ax.plot(X_coordinate, Y_coordinate,
                        marker='o', fillstyle='none')
                ax.annotate(sname, xy=(X_coordinate, Y_coordinate), xytext=(
                    3, 3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                # starlist.append((sname,X_coordinate,Y_coordinate))
                ax.figure.canvas.draw()

                cldim = self.cloudimage
                sr = StarReference(sname, [X_coordinate, Y_coordinate])
                Ok = False
                try:
                    sr.getSkyCoord()
                    Ok = True
                except Exception as e:
                    print(e)
                    Ok = gui_string(
                        caption='Neatpazīst zvaigzni, vai ievadīt?') is not None
                if Ok:
                    cldim.starReferences.append(sr)

        else:
            self.StopCiparotZvaigznes()

    def StartCiparotZvaigznes(self):
        if self.isCiparotZvaigznes is None:
            self.MplWidget1.canvas.fig.set_facecolor('mistyrose')
            self.ZimetAttelu()
            self.isCiparotZvaigznes = self.MplWidget1.canvas.mpl_connect(
                'button_press_event', self.onclick_ciparotzvaigznes)

    def StopCiparotZvaigznes(self):
        if self.isCiparotZvaigznes is not None:
            self.MplWidget1.canvas.mpl_disconnect(self.isCiparotZvaigznes)
            filename_stars = os.path.splitext(self.cloudimage.filename)[0]+'_zvaigznes.txt'
            filename_stars, _ = QFileDialog.getSaveFileName(directory=filename_stars,
                                                       filter='(*.txt)',
                                                       caption='Zvaigžņu fails')
            if filename_stars != '':            
                self.cloudimage.saveStarReferences(filename_stars)
            self.MplWidget1.canvas.fig.set_facecolor('white')
            self.ZimetAttelu()
        self.isCiparotZvaigznes = None

    def CiparotZvaigznesClick(self):
        if self.isCiparotZvaigznes is None:
            self.StartCiparotZvaigznes()
        else:
            self.StopCiparotZvaigznes()

    def ProjicetClick(self, tips=0):
        if hasattr(self.cloudimage, "camera"):
            text = f'{self.projHeight}'
            s = gui_string(text=text, caption='Augstums kilometros')
            if s is not None:
                try:
                    self.projHeight = float(s)
                    if tips in [0, 1]:
                        self.Projicet(self.projHeight, atseviski=tips == 0)
                    else:
                        self.ProjicetVidejotuAttelu(self.projHeight)
                except:
                    print('Nepareiza ievade!')
                    raise

    def ProjicetNoKartesClick(self, tips=0):
        if hasattr(self.cloudimage, "camera") and self.heightmap is not None:
            if tips in [0, 1]:
                self.Projicet(self.heightmap.heightmap /
                              1000.0, atseviski=tips == 0)
            else:
                self.ProjicetVidejotuAttelu(self.heightmap.heightmap/1000.0)

    def distance(self, p1_webmerc, p2_webmerc):
        ll1 = tilemapbase.to_lonlat(*p1_webmerc)
        ll2 = tilemapbase.to_lonlat(*p2_webmerc)
        import pymap3d.vincenty
        # reverse lonlat to latlon form pymap3d.vincenty
        ll1=ll1[1],ll1[0]
        ll2=ll2[1],ll2[0]
        dist, az = pymap3d.vincenty.vdist(*ll1, *ll2)
        return dist

    def onclick_meritattalumu(self, event):
        ax = event.inaxes
        if ax is None:
            return
        # print(event)
        try:  # use try/except in case we are not using Qt backend
            # 0 is the arrow, which means we are not zooming or panning.
            zooming_panning = (
                ax.figure.canvas.cursor().shape() not in [0, 13])
        except:
            zooming_panning = False
        if zooming_panning:
            #print("Zooming or panning")
            return
        #print('you pressed', event.key, event.xdata, event.ydata)
        if event.button == 1:
            if self.meerit is None:
                # print(self.meerit)
                pp = ax.plot(event.xdata, event.ydata, marker='D',ms=5, color='orange')
                self.meerit = {'p1':(event.xdata, event.ydata), 'plotp1':pp}
                ax.figure.canvas.draw()
            else:
                try:
                    xx = (event.xdata, event.ydata)
                    #print(self.meerit, xx)
                    dist = self.distance(self.meerit['p1'], xx)
                    for c in ['plotp1','ln']:
                        if c in self.meerit:
                            line = self.meerit[c][0]
                            line.remove() 
                    if 'annot' in self.meerit:
                        self.meerit['annot'].remove()                            
                    self.meerit = None
                    print(f'{dist:.0f}m')
                    ax.figure.canvas.draw()
                except:
                    raise

    def move_meritattalumu(self, event):
        if self.meerit is not None:
            ax = event.inaxes
            if ax is not None:
                xx = (event.xdata, event.ydata)
                dist = self.distance(self.meerit['p1'], xx)
                
                if 'ln' in self.meerit:
                    line = self.meerit['ln'].pop(0)
                    line.remove()                                
                if 'annot' in self.meerit:
                    self.meerit['annot'].remove()
                self.meerit['ln'] = ax.plot([self.meerit['p1'][0], xx[0]], [self.meerit['p1'][1], xx[1]], color='black')
                self.meerit['annot']=ax.annotate(f'{dist:.0f}m',xy=xx, xytext=(10,10), textcoords='offset points', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
                ax.figure.canvas.draw()

    def plotProjicet(self, pimages, ax, plotMap=True, plotPoints=True):
        def xy_latlon_str(x, y):
            lon, lat = tilemapbase.to_lonlat(x, y)
            return f'{lat:.3f}, {lon:.3f}'
        ax.format_coord = xy_latlon_str

        self.meerit = None
        pp = [[self.cloudimage.location.lon.value,
               self.cloudimage.location.lat.value]]
        if self.cloudimage2 is not None:
            pp.append([self.cloudimage2.location.lon.value,
                      self.cloudimage2.location.lat.value])
        plots.PlotReferencedImages(self.webmerc, pimages,
                                   camera_points=pp if plotPoints else [],
                                   outputFileName=None,
                                   lonmin=self.map_bounds[0], lonmax=self.map_bounds[
                                       1], latmin=self.map_bounds[2], latmax=self.map_bounds[3],
                                   alpha=self.map_alpha,
                                   ax=ax,
                                   plotMap=plotMap)
    def disconnect_meerit(self):
        if self.meerit_events is not None:
            for e in self.meerit_events:
                self.MplWidget1.canvas.mpl_disconnect(e)
            self.meerit_events=None
    def connect_meerit(self):
        self.meerit_events = [self.MplWidget1.canvas.mpl_connect(
            'button_press_event', self.onclick_meritattalumu),
        self.MplWidget1.canvas.mpl_connect(
            'motion_notify_event', self.move_meritattalumu)]
            
    def Projicet(self, projHeight, atseviski=True):
        
        self.disconnect_meerit()
        
        self.webmerc.cloudImage = self.cloudimage
        self.webmerc.prepare_reproject_from_camera(projHeight)
        projected_image = self.webmerc.Fill_projectedImageMasked()
        self.projected_image = (
            self.webmerc.__getstate__(), projected_image, projHeight)
        pimages = [projected_image]
        if self.cloudimage2 is not None:
            self.webmerc.cloudImage = self.cloudimage2
            self.webmerc.prepare_reproject_from_camera(projHeight)
            projected_image2 = self.webmerc.Fill_projectedImageMasked()
            pimages.append(projected_image2)
            self.webmerc.cloudImage = self.cloudimage

        if atseviski and self.cloudimage2 is not None:
            self.MplWidget1.canvas.initplot([121, 122])
            self.plotProjicet(pimages[0:1], self.MplWidget1.canvas.ax[0])
            self.plotProjicet(pimages[1:2], self.MplWidget1.canvas.ax[1])
        else:
            self.MplWidget1.canvas.initplot()
            self.plotProjicet(pimages, self.MplWidget1.canvas.ax)

        self.connect_meerit()
        self.MplWidget1.canvas.draw()
        self.pelekot()

    def ProjicetVidejotuAttelu(self, projHeight):
        self.disconnect_meerit()
        
        if self.cloudimage2 is not None:
            self.webmerc.cloudImage = self.cloudimage
            self.webmerc.prepare_reproject_from_camera(projHeight)
            projected_image = self.webmerc.Fill_projectedImage()
            pimages = [projected_image]
            self.webmerc.cloudImage = self.cloudimage2
            self.webmerc.prepare_reproject_from_camera(projHeight)
            projected_image2 = self.webmerc.Fill_projectedImage()
            pimages.append(projected_image2)
            self.webmerc.cloudImage = self.cloudimage
            img_mean, img_diff, _, img_bicolor = utils.getAverageImages(
                pimages)
            self.MplWidget1.canvas.initplot([131, 132, 133])
            self.plotProjicet(
                [img_mean], self.MplWidget1.canvas.ax[0], plotMap=False, plotPoints=False)
            self.plotProjicet(
                [img_diff], self.MplWidget1.canvas.ax[1], plotMap=False, plotPoints=False)
            self.plotProjicet(
                [img_bicolor[(0, 1)]], self.MplWidget1.canvas.ax[2], plotMap=False, plotPoints=False)
            self.MplWidget1.canvas.draw()
            self.connect_meerit()
    def MainitApgabalu(self):
        w = self.webmerc
        text = f'{w.lonmin},{w.lonmax},{w.latmin},{w.latmax},{w.pixel_per_km}'
        s = gui_string(
            text=text, caption='lonmin,lonmax,latmin,latmax,resolution_km')
        if s is not None:
            try:
                s = [float(x) for x in s.split(',')]
                if len(s) == 5:
                    self.webmerc = WebMercatorImage(self.cloudimage, *s)
                    print(self.webmerc)
            except:
                print('Nepareiza ievade!')
                pass

    def KartesApgabals(self):
        w = self.map_bounds
        text = f'{w[0]},{w[1]},{w[2]},{w[3]},{self.map_alpha}'
        s = gui_string(
            text=text, caption='lonmin,lonmax,latmin,latmax,map_alpha')
        if s is not None:
            try:
                s = [float(x) for x in s.split(',')]
                if len(s) == 5:
                    self.map_bounds = s[:4]
                    self.map_alpha = max(min(s[4], 1.0), 0.0)
            except:
                print('Nepareiza ievade!')
                pass

    def onclick_ciparotkontrolpunktus(self, event):
        # https://stackoverflow.com/a/64486726
        ax = event.inaxes
        if ax is None:
            return
        try:  # use try/except in case we are not using Qt backend
            # 0 is the arrow, which means we are not zooming or panning.
            zooming_panning = (
                ax.figure.canvas.cursor().shape() not in [0, 13])
        except:
            zooming_panning = False
        if zooming_panning:
            return
        # print(event)
        if event.button != 1:
            self.StopCiparotKontrolpunkti()

        #print(self.pairNo, ax == self.MplWidget1.canvas.ax[0],ax == self.MplWidget1.canvas.ax[1])
        if not ((ax == self.MplWidget1.canvas.ax[0] and self.pairNo == 0) or (ax == self.MplWidget1.canvas.ax[1] and self.pairNo == 1)):
            return
        if event.button == 1:
            X_coordinate = event.xdata
            Y_coordinate = event.ydata
            self.cpair.correspondances[self.pairNo] = np.append(
                self.cpair.correspondances[self.pairNo], [[X_coordinate, Y_coordinate]], axis=0)
            num = len(self.cpair.correspondances[self.pairNo])

            ax.plot(X_coordinate, Y_coordinate, marker='o',
                    fillstyle='none', markeredgecolor='red')
            ax.annotate(str(num), xy=(X_coordinate, Y_coordinate), xytext=(
                3, 3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
            # starlist.append((sname,X_coordinate,Y_coordinate))

            if self.pairNo == 0:
                epilines = self.cpair.GetEpilinesAtHeightInterval(
                    [75, 90], [[X_coordinate, Y_coordinate]], True)
                # print(epilines)
                self.tempepiline = self.MplWidget1.canvas.ax[1].plot(epilines[0, :, 0], epilines[0, :, 1],
                                                                     color='yellow', marker='o', ms=1, lw=0.8)
            if self.pairNo == 1:
                llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                    [self.cpair.correspondances[0][-1]], [self.cpair.correspondances[1][-1]])
                try:
                    print(
                        f'Augstums {llh[2][0]/1000.0:.1f}km, Staru attālums {rayminimaldistance[0]:.1f}m')
                    line = self.tempepiline.pop(0)
                    line.remove()
                except:
                    pass

            ax.figure.canvas.draw()

            self.pairNo = (self.pairNo + 1) % 2

    def CiparotKontrolpunktusClick(self):
        
        if self.isCiparotKontrolpunkti is None:
            self.StartCiparotKontrolpunkti()
        else:
            self.StopCiparotKontrolpunkti()

    def StartCiparotKontrolpunkti(self):
        if self.isCiparotKontrolpunkti is None and self.cloudimage2 is not None:
            self.MplWidget1.canvas.fig.set_facecolor('plum')
            self.ZimetAttelu(otrs=True, kontrolpunkti=True)
            if self.cpair is None:
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
            self.pairNo = 0
            self.isCiparotKontrolpunkti = self.MplWidget1.canvas.mpl_connect(
                'button_press_event', self.onclick_ciparotkontrolpunktus)

    def StopCiparotKontrolpunkti(self):
        if self.isCiparotKontrolpunkti is not None:
            self.MplWidget1.canvas.mpl_disconnect(self.isCiparotKontrolpunkti)
            ll = min(len(self.cpair.correspondances[0]), len(
                self.cpair.correspondances[1]))
            self.cpair.correspondances[0] = self.cpair.correspondances[0][0:ll]
            self.cpair.correspondances[1] = self.cpair.correspondances[1][0:ll]
            matchfile = f'{os.path.split(self.cloudimage.filename)[0]}/{self.cloudimage.code}_{self.cloudimage2.code}.txt'
            matchfile, _ = QFileDialog.getSaveFileName(directory=matchfile,
                                                       filter='(*.txt)',
                                                       caption='Atbilstību fails')
            if matchfile != '':
                self.cpair.SaveCorrespondances(matchfile)
            self.MplWidget1.canvas.fig.set_facecolor('white')
            self.ZimetAttelu(otrs=True, kontrolpunkti=True)
        self.isCiparotKontrolpunkti = None
        self.pelekot()

    def IelasitKontrolpunktus(self):
        if self.cloudimage2 is not None:
            matchfile = f'{os.path.split(self.cloudimage.filename)[0]}/{self.cloudimage.code}_{self.cloudimage2.code}.txt'
            matchfile = gui_fname(directory=matchfile, filter='(*.txt)')
            if matchfile != '':
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
                self.cpair.LoadCorrespondances(matchfile)
                self.ZimetAttelu(otrs=True, kontrolpunkti=True)
        self.pelekot()

    def plotMatches(self, ax, pairNo):
        if self.cpair is not None:
            for i, (x, y) in enumerate(self.cpair.correspondances[pairNo]):
                ax.plot(x, y, marker='o', fillstyle='none',
                        markeredgecolor='red')
                ax.annotate(str(i+1), xy=(x, y), xytext=(3, 3),
                            color='#AAFFAA', fontsize=16, textcoords='offset pixels')

    def ZimetKontrolpunktuAugstumus(self):
        self.disconnect_meerit()
        
        if self.cpair is not None:
            self.MplWidget1.canvas.initplot([121, 122])
            ax = self.MplWidget1.canvas.ax[0]
            ax2 = self.MplWidget1.canvas.ax[1]

            z1, z2 = 75, 90

            llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            # epilīnijas pirmajā attēlā, kas atbilst punktiem otrajā attēlā
            epilines = self.cpair.GetEpilinesAtHeightInterval(
                [z1, z2], self.cpair.correspondances[1], False)
            plots.PlotValidHeightPoints(self.cloudimage.imagearray, epilines, self.cpair.correspondances[0], llh[2], None,
                                        ax=ax)
            # epilīnijas otrajā attēlā, kas atbilst punktiem pirmajā attēlā
            epilines = self.cpair.GetEpilinesAtHeightInterval(
                [z1, z2], self.cpair.correspondances[0], True)
            plots.PlotValidHeightPoints(self.cloudimage2.imagearray, epilines, self.cpair.correspondances[1], llh[2], None,
                                        ax=ax2)
            self.MplWidget1.canvas.draw()

    def IzveidotAugstumuKarti(self):
        if self.cpair is not None:
            llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            if any(valid):
                self.webmerc.cloudimage = self.cloudimage
                heightgrid = self.webmerc.PrepareHeightMap(
                    llh[1][valid], llh[0][valid], llh[2][valid])
                self.heightmap = HeightMap(self.webmerc)
                self.heightmap.heightmap = heightgrid
                self.heightmap.points = llh
                self.heightmap.validpoints = valid
                self.ZimetAugstumuKarti()
            else:
                print('Nevar izveidot augstumu karti - nav derīgu kontrolpunktu')
            self.pelekot()

    def SaglabatAugstumuKarti(self):
        if self.heightmap is not None:
            projfile = os.path.splitext(self.cloudimage.filename)[0]+'.hmp'
            projfile, _ = QFileDialog.getSaveFileName(directory=projfile,
                                                      filter='(*.hmp)',
                                                      caption='Augstumu kartes fails')
            if projfile != '':
                self.heightmap.save(projfile)
                print(f'Saved heightmap {projfile}')

    def IelasitAugstumuKarti(self):
        if self.cloudimage is not None:
            projfile = os.path.splitext(self.cloudimage.filename)[0]+'.hmp'
            projfile, _ = QFileDialog.getOpenFileName(directory=projfile,
                                                      filter='(*.hmp)',
                                                      caption='Augstumu kartes fails')
            if projfile != '':
                self.heightmap = HeightMap.load(projfile)
                self.webmerc = self.heightmap.webmerc
                print(f'Saved heightmap {projfile}')
            self.pelekot()

    def ZimetAugstumuKarti(self):
        self.disconnect_meerit()
        
        if self.heightmap is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            csl = plots.PlotReferencedImages(self.webmerc, [self.heightmap.heightmap],  camera_points=[],
                                             outputFileName=None,
                                             lonmin=self.map_bounds[0], lonmax=self.map_bounds[
                                                 1], latmin=self.map_bounds[2], latmax=self.map_bounds[3],
                                             showplot=True,
                                             alpha=0.8, ax=ax)
            import tilemapbase
            llh = self.heightmap.points
            valid = self.heightmap.validpoints
            xy = np.array([tilemapbase.project(lon, lat)
                          for lon, lat in zip(llh[1][valid], llh[0][valid])])
            cs = ax.scatter(xy[:, 0], xy[:, 1], c=llh[2]
                            [valid], norm=csl[0].norm, cmap=csl[0].cmap)
            ax.figure.colorbar(csl[0])
            self.connect_meerit()
            self.MplWidget1.canvas.draw()

    def KamerasKalibracijasParametri(self):
        s = f'{int(self.camera_calib_params["distortion"])},{int(self.camera_calib_params["centers"])},{int(self.camera_calib_params["separate_x_y"])},{self.camera_calib_params["projectiontype"]}'
        s = gui_string(text=s, caption="distortion,centers,separate_x_y,projectiontype")
        if s is not None:
            try:
                #s = [int(x) for x in s.split(',')]
                s=s.split(',')
                self.camera_calib_params["distortion"] = int(s[0]) == 1
                self.camera_calib_params["centers"] = int(s[1]) != 0
                self.camera_calib_params["separate_x_y"] = int(s[2]) != 0
                self.camera_calib_params["projectiontype"] = s[3]
                print('Kalibrēšanas parametri:', self.camera_calib_params)
            except Exception as e:
                print('Nepareizi ievades parametri',e)

    def SaglabatProjicetoAttelu(self, jpg=True):
        if self.projected_image is not None:
            ext = '.jpg' if jpg else '.tif'
            extjgw = '.jgw' if jpg else '.tfw'
            f = os.path.split(self.cloudimage.filename)
            try:
                z = float(self.projected_image[2])
                zs = f'_{z:.1f}'
            except:
                zs = ''

            projfile = f[0]+'/proj_'+os.path.splitext(f[1])[0]+zs+ext

            projfile, _ = QFileDialog.getSaveFileName(directory=projfile,
                                                      filter=f'(*{ext})',
                                                      caption='Projicētais attēls')
            if projfile != '':
                jgwfile = os.path.splitext(projfile)[0]+extjgw
                if jpg:
                    img = self.projected_image[1][:, :, 0:3]
                else:
                    img = self.projected_image[1]
                wm = WebMercatorImage(None, 17, 33, 56, 63, 1.0)
                wm.__setstate__(self.projected_image[0])
                wm.SaveJgw(jgwfile)
                import skimage.io
                skimage.io.imsave(projfile, img)
                if not jpg:
                    from sudrabainiemakoni import savekml
                    savekml.mapOverlay(wm, img, self.projHeight, projfile, saveimage=False, cloudimage=self.cloudimage)
                print('Fails saglabāts:', projfile)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    myapp = MainW()
    myapp.show()
    sys.exit(app.exec_())
