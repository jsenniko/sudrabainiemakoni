import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui, QtCore
#from matplotlib.backend_tools import Cursors
import tilemapbase
from sudrabainiemakoni.cloudimage import CloudImage, CloudImagePair, HeightMap
from sudrabainiemakoni.cloudimage_camera import CameraCalibrationParams
from guihelpers.camera_parameters import show_camera_parameters_dialog
from guihelpers.camera_modification import show_camera_modification_dialog
from guihelpers.settings import AppSettings
from sudrabainiemakoni.webmercatorimage import ProjectionImageWebMercator
from sudrabainiemakoni.starreference import StarReference
from sudrabainiemakoni import cameraprojections, plots, utils
from smgui import Ui_MainWindow
from qthelper import gui_fname, gui_save_fname, gui_string
import smhelper
from exceptions import handle_exceptions
from star_digitizer import StarDigitizer
from control_point_digitizer import ControlPointDigitizer


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

        self.actionIelas_t_att_lu.triggered.connect(self.LoadImageClick)
        self.actionMainit_att_lu.triggered.connect(self.ChangeImageClick)
        self.actionKalibr_t_kameru.triggered.connect(self.CalibrateCameraClick)
        self.actionSaglab_t_projektu.triggered.connect(self.SaveProject)
        self.actionIelas_t_projektu.triggered.connect(self.LoadProject)
        self.actionIelas_t_otro_projektu.triggered.connect(
            self.LoadProject2)
        self.actionHorizont_lo_koordin_tu_re_is.triggered.connect(
            self.DrawAltAzClick)
        self.actionAtt_lu.triggered.connect(self.DrawImageClick)
        self.actionCiparot_zvaigznes.triggered.connect(
            self.DigitizeStarsClick)
        self.actionProjic_t.triggered.connect(
            lambda: self.ProjectClick(tips=0))
        self.actionProjic_t_kop.triggered.connect(
            lambda: self.ProjectClick(tips=2))
        self.actionProjic_t_no_augstumu_kartes.triggered.connect(
            lambda: self.ProjectFromMapClick(tips=0))
        self.actionProjic_t_kop_no_augstumu_kartes.triggered.connect(
            lambda: self.ProjectFromMapClick(tips=2))
        self.actionProjekcijas_apgabals.triggered.connect(self.ChangeRegion)
        self.actionKartes_apgabals.triggered.connect(self.MapRegion)
        self.actionKontrolpunkti.triggered.connect(
            self.DigitizeControlPointsClick)
        self.actionIelas_t_kontrolpunktus.triggered.connect(
            self.LoadControlPoints)
        self.actionSaglab_t_augstumu_punktus_CSV.triggered.connect(
            self.SaveHeightPointsCSV)
        self.actionKontrolpunktu_augstumus.triggered.connect(
            self.DrawControlPointHeights)
        self.actionIzveidot_augstumu_karti.triggered.connect(
            self.CreateHeightMap)
        self.actionIelas_t_augstumu_karti.triggered.connect(
            self.LoadHeightMap)
        self.actionSaglab_t_augstumu_karti.triggered.connect(
            self.SaveHeightMap)
        self.actionAugstumu_karti.triggered.connect(self.DrawHeightMap)
        self.actionKameras_kalibr_cijas_parametri.triggered.connect(
            self.CameraCalibrationParameters)
        self.actionKameras_modifikacija.triggered.connect(
            self.CameraModification)
        self.actionSaglab_t_projic_to_att_lu_JPG.triggered.connect(
            lambda: self.SaveProjectedImage(jpg=True))
        self.actionSaglab_t_projic_to_att_lu_TIFF.triggered.connect(
            lambda: self.SaveProjectedImage(jpg=False))
        self.actionIelas_t_kameru.triggered.connect(self.LoadCamera)
        self.actionSaglab_t_kameru.triggered.connect(self.SaveCamera)
        self.actionUzst_d_t_datumu.triggered.connect(self.SetDate)
        self.actionUzst_d_t_platumu_garumu_augstumu.triggered.connect(self.SetCoordinates)
        self.actionEpil_niju_augstums.triggered.connect(self.SetEpilineHeight)
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        self.cloudimage = None
        self.cloudimage2 = None
        self.cpair = None
        self.webmerc = ProjectionImageWebMercator(self.cloudimage, 17, 33, 56, 63, 1.0)
        self.projHeight = 80  # km
        self.map_bounds = [17, 33, 56, 63]
        self.map_alpha = 0.85
        self.heightmap = None
        self.projected_image = None
        # Create local AppSettings instance
        self.app_settings = AppSettings()
        
        # Load settings into UI components
        self.load_settings_to_ui()

        self.isDigitizeStars = None
        self.isDigitizeControlPoints = None
        self.measure_events = None
        self.z1, self.z2 = 75, 90
        
        # Initialize digitizers (will be created when needed)
        self.star_digitizer = None
        self.control_point_digitizer = None

    def load_settings_to_ui(self):
        """Load settings from app_settings into UI components"""
        # Set file dialog manager's last directory from settings
        from qthelper import file_dialog_manager
        if self.app_settings.last_directory:
            file_dialog_manager.last_directory = self.app_settings.last_directory

    def save_ui_to_settings(self):
        """Save UI state to app_settings and persist to file"""
        # Save last directory from file dialog manager to settings
        from qthelper import file_dialog_manager
        self.app_settings.last_directory = file_dialog_manager.last_directory
        self.app_settings.save_to_file()

    def update_ui_state(self):
        self.actionMainit_att_lu.setEnabled(self.cloudimage is not None)
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
        self.actionSaglab_t_augstumu_punktus_CSV.setEnabled(self.cpair is not None)
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
        self.actionKameras_modifikacija.setEnabled(self.cloudimage is not None)
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
        # Save UI state to settings
        self.save_ui_to_settings()
        
        # Return standard output to defaults.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

    def LoadImageClick(self):
        filename_jpg = gui_fname(
            caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg != '':
            self.LoadImage(filename_jpg)

    @handle_exceptions(method_name="Loading image")
    def LoadImage(self, filename_jpg):
        self.cloudimage2 = None
        self.console.clear()
        case_id = os.path.splitext(os.path.split(filename_jpg)[1])[0]
        filename_stars = smhelper.check_stars_file(filename_jpg)
        lat, lon, height = smhelper.check_latlon_file(filename_jpg)
        self.cloudimage = CloudImage.from_files(
            case_id, filename_jpg, filename_stars, lat, lon, height=height)
        # Reset digitization state and update star digitizer with new cloudimage
        self.isDigitizeStars = None
        self.DrawImage()
        self.update_ui_state()

    def ChangeImageClick(self):
        filename_jpg = gui_fname(
            caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg != '':
            self.ChangeImage(filename_jpg)
    @handle_exceptions(method_name="Changing image")
    def ChangeImage(self, filename_jpg):
        if self.cloudimage is not None:
            self.cloudimage.filename = filename_jpg
            self.cloudimage.LoadImage(reload=True)
        self.update_ui_state()

            
    @handle_exceptions(method_name="Setting date")
    def SetDate(self):
        d=self.cloudimage.date.to_datetime(timezone=CloudImage.timezone)
        s=d.strftime('%Y-%m-%dT%H:%M:%S')
        s=gui_string(text=s, caption='Ievadi datumu, YYYY-MM-DDTHH:MM:SS')
        if s is not None:
            import datetime
            d=datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            self.cloudimage.setDate(d)
            
    @handle_exceptions(method_name="Setting coordinates")
    def SetCoordinates(self):
        lat,lon,height=self.cloudimage.getLocation()
        s=f'{lat:.5f},{lon:.5f},{height:.0f}'
        s=gui_string(text=s, caption='Ievadi koordinātes (lat,lon,z)')
        if s is not None:
            s=s.split(',')   
            lat,lon,height=float(s[0]), float(s[1]),0.0
            if len(s)>2:
                height=float(s[2])
            self.cloudimage.setLocation(lon=lon, lat=lat, height=height)
            
    @handle_exceptions(method_name="Setting epiline height")
    def SetEpilineHeight(self):
        s=f'{self.z1:.0f},{self.z2:.0f}'
        s=gui_string(text=s, caption='Ievadi z1,z2')
        if s is not None:
            s=s.split(',')   
            self.z1,self.z2=float(s[0]), float(s[1])
            # Update control point digitizer if active
            if self.control_point_digitizer is not None:
                self.control_point_digitizer.set_height_range(self.z1, self.z2)
    def PrintCameraParameters(self):
        if self.cloudimage is not None:
            cldim = self.cloudimage
            az, el, rot = cldim.camera.get_azimuth_elevation_rotation()
            print(f'Kameras ass azimuts {az:.2f}°')
            print(f'Kameras ass augstums virs horizonta {el:.2f}°')
            print(f'Kameras pagrieziena leņķis {rot:.2f}°')
            fx, fy, cx, cy = cldim.camera.get_focal_lengths_mm()
            print(
                f'Kameras fokusa attālumi (35mm ekvivalents) {fx:.1f} {fy:.1f}')
            print(f'Kameras ass pozīcija {cx:.1f} {cy:.1f}')
            print(f'Distortion: k1={cldim.camera.camera_enu.k1:.6f}, k2={cldim.camera.camera_enu.k2:.6f}, k3={cldim.camera.camera_enu.k3:.6f}')
            print(f'Projection: {cameraprojections.name_by_projection(cldim.camera.camera_enu.projection)}')
            print(f'Projection: {type(cldim.camera.camera_enu.projection)}')

    @handle_exceptions(method_name="Camera calibration")
    def CalibrateCameraClick(self):
        if self.cloudimage is not None:
            cldim = self.cloudimage
            cldim.PrepareCamera(**self.app_settings.camera_calibration.to_dict())
            # distortion=False, centers=True, separate_x_y=True
            #cldim.PrepareCamera(method='optnew', distortion=args.optimizeDistortion, centers=args.notOptimizeCenter, separate_x_y=args.notOptimizeUnsymmetric)
            self.PrintCameraParameters()
            self.DrawAltAzClick()
        self.update_ui_state()

    @handle_exceptions(method_name="Loading project")
    def LoadProject(self):
        projfile = gui_fname(caption='Projekta fails', filter='(*.proj)')
        if projfile != '':
            print(f'Loading project {projfile}')
            self.cloudimage2 = None
            self.cpair = None
            self.console.clear()
            self.cloudimage = CloudImage.load(projfile)
            # Reset digitization state and update star digitizer with new cloudimage
            self.isDigitizeStars = None
            print(f'Loaded project file {projfile}')
            print(self.cloudimage)
            self.DrawImage()
        self.update_ui_state()
            
    @handle_exceptions(method_name="Loading second project")
    def LoadProject2(self):
        projfile = gui_fname(caption='Projekta fails', filter='(*.proj)')
        if projfile != '':
            self.cpair = None
            self.cloudimage2 = CloudImage.load(projfile)
            print(f'Loaded project file {projfile}')
            print(self.cloudimage2)
            self.DrawImage(otrs=True)
        self.update_ui_state()

    @handle_exceptions(method_name="Saving project")
    def SaveProject(self):
        projfile = os.path.splitext(self.cloudimage.filename)[0]+'.proj'
        projfile = gui_save_fname(
            directory=projfile,
            caption='Projekta fails',
            filter='(*.proj)')
        if projfile != '':
            self.cloudimage.save(projfile)
            print(f'Saved project file {projfile}')
            
    @handle_exceptions(method_name="Loading camera")
    def LoadCamera(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0]+'_enu.json'
        camfile = gui_fname(
            directory=camfile,
            caption='Kameras fails',
            filter='(*.json)')
        if camfile !='':
            self.cloudimage.LoadCamera(camfile)
            
    @handle_exceptions(method_name="Saving camera")
    def SaveCamera(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0]+'_enu.json'
        camfile = gui_save_fname(
            directory=camfile,
            caption='Kameras fails',
            filter='(*.json)')
        if camfile !='':
            self.cloudimage.SaveCamera(camfile)
        
    @handle_exceptions(method_name="Drawing Alt-Az grid")
    def DrawAltAzClick(self):
        self.disconnect_measurement()
        if self.cloudimage is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            plots.PlotAltAzGrid(self.cloudimage, ax=ax)
            cldim = self.cloudimage
            # zvaigžņu koordinātes enu sistēmā, vienības attālumam
            enu_unit_coords = cldim.get_stars_enu_unit_coords()
            if len(enu_unit_coords)>0:
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

    @handle_exceptions(method_name="Drawing image")
    def DrawImageClick(self):
        self.DrawImage(otrs=self.cloudimage2 is not None)

    @handle_exceptions(method_name="Drawing image")
    def DrawImage(self, otrs=False, kontrolpunkti=False, plot_stars = True):
        self.disconnect_measurement()
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
                self.plot_matches(ax, 0)
            else:
                if plot_stars:
                    plots.PlotStars(self.cloudimage, ax)
            if otrs and self.cloudimage2 is not None:
                ax2.imshow(self.cloudimage2.imagearray)
                if kontrolpunkti:
                    self.plot_matches(ax2, 1)
                else:
                    if plot_stars:
                        plots.PlotStars(self.cloudimage2, ax2)

            self.MplWidget1.canvas.draw()
            return ax
            
    # onclick_digitize_stars method now handled by StarDigitizer

    @handle_exceptions(method_name="Starting star digitization")
    def StartDigitizeStars(self):
        if self.isDigitizeStars is None:     
            self.DrawImage(plot_stars = False)       
            # Get current axes from canvas
            ax = self.MplWidget1.canvas.ax
            if hasattr(ax, '__len__'):  # Multiple axes
                ax = ax[0]  # Use first axis
            
            # Create star digitizer with current axes and cloudimage
            self.star_digitizer = StarDigitizer(ax, self.cloudimage, self)
            self.star_digitizer.start_digitization()
            self.isDigitizeStars = True  # Keep for compatibility

    @handle_exceptions(method_name="Stopping star digitization")
    def StopDigitizeStars(self):
        if self.isDigitizeStars is not None and self.star_digitizer is not None:
            self.star_digitizer.stop_digitization()
            self.star_digitizer = None
        self.isDigitizeStars = None

    @handle_exceptions(method_name="Star digitization button click")
    def DigitizeStarsClick(self):
        if self.isDigitizeStars is None:
            self.StartDigitizeStars()
        else:
            self.StopDigitizeStars()

    @handle_exceptions(method_name="Projection click")
    def ProjectClick(self, tips=0):
        if hasattr(self.cloudimage, "camera"):
            text = f'{self.projHeight}'
            s = gui_string(text=text, caption='Augstums kilometros')
            if s is not None:
                try:
                    self.projHeight = float(s)
                    if tips in [0, 1]:
                        self.Project(self.projHeight, atseviski=tips == 0)
                    else:
                        self.ProjectAveragedImage(self.projHeight)
                except Exception as e:
                    print(f'Nepareiza ievade! {str(e)}')
                    raise

    @handle_exceptions(method_name="Projection from height map click")
    def ProjectFromMapClick(self, tips=0):
        if hasattr(self.cloudimage, "camera") and self.heightmap is not None:
            if tips in [0, 1]:
                self.Project(self.heightmap.heightmap /
                              1000.0, atseviski=tips == 0)
            else:
                self.ProjectAveragedImage(self.heightmap.heightmap/1000.0)

    @handle_exceptions(method_name="Calculating distance")
    def distance(self, p1_webmerc, p2_webmerc):
        ll1 = tilemapbase.to_lonlat(*p1_webmerc)
        ll2 = tilemapbase.to_lonlat(*p2_webmerc)
        import pymap3d.vincenty
        # reverse lonlat to latlon form pymap3d.vincenty
        ll1=ll1[1],ll1[0]
        ll2=ll2[1],ll2[0]
        dist, az = pymap3d.vincenty.vdist(*ll1, *ll2)
        return dist

    @handle_exceptions(method_name="Distance measurement click")
    def onclick_measure_distance(self, event):
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
            if self.measure is None:
                # print(self.measure)
                pp = ax.plot(event.xdata, event.ydata, marker='D',ms=5, color='orange')
                self.measure = {'p1':(event.xdata, event.ydata), 'plotp1':pp}
                ax.figure.canvas.draw()
            else:
                xx = (event.xdata, event.ydata)
                #print(self.measure, xx)
                dist = self.distance(self.measure['p1'], xx)
                for c in ['plotp1','ln']:
                    if c in self.measure:
                        line = self.measure[c][0]
                        line.remove() 
                if 'annot' in self.measure:
                    self.measure['annot'].remove()                            
                self.measure = None
                print(f'{dist:.0f}m')
                ax.figure.canvas.draw()

    @handle_exceptions(method_name="Distance measurement movement")
    def move_measure_distance(self, event):
        if self.measure is not None:
            ax = event.inaxes
            if ax is not None:
                xx = (event.xdata, event.ydata)
                dist = self.distance(self.measure['p1'], xx)
                
                if 'ln' in self.measure:
                    line = self.measure['ln'].pop(0)
                    line.remove()                                
                if 'annot' in self.measure:
                    self.measure['annot'].remove()
                self.measure['ln'] = ax.plot([self.measure['p1'][0], xx[0]], [self.measure['p1'][1], xx[1]], color='black')
                self.measure['annot']=ax.annotate(f'{dist:.0f}m',xy=xx, xytext=(10,10), textcoords='offset points', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
                ax.figure.canvas.draw()

    @handle_exceptions(method_name="Plotting projection")
    def plot_projection(self, pimages, ax, plotMap=True, plotPoints=True):
        def xy_latlon_str(x, y):
            lon, lat = tilemapbase.to_lonlat(x, y)
            return f'{lat:.3f}, {lon:.3f}'
        ax.format_coord = xy_latlon_str

        self.measure = None
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
            
    @handle_exceptions(method_name="Disconnecting measurement")
    def disconnect_measurement(self):
        if self.measure_events is not None:
            for e in self.measure_events:
                self.MplWidget1.canvas.mpl_disconnect(e)
            self.measure_events=None
            
    @handle_exceptions(method_name="Connecting measurement")
    def connect_measurement(self):
        self.measure_events = [self.MplWidget1.canvas.mpl_connect(
            'button_press_event', self.onclick_measure_distance),
        self.MplWidget1.canvas.mpl_connect(
            'motion_notify_event', self.move_measure_distance)]
            
    @handle_exceptions(method_name="Projecting image")
    def Project(self, projHeight, atseviski=True):
        self.disconnect_measurement()
        
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
            self.plot_projection(pimages[0:1], self.MplWidget1.canvas.ax[0])
            self.plot_projection(pimages[1:2], self.MplWidget1.canvas.ax[1])
        else:
            self.MplWidget1.canvas.initplot()
            self.plot_projection(pimages, self.MplWidget1.canvas.ax)

        self.connect_measurement()
        self.MplWidget1.canvas.draw()
        self.update_ui_state()

    @handle_exceptions(method_name="Projecting averaged image")
    def ProjectAveragedImage(self, projHeight):
        self.disconnect_measurement()
        
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
            self.plot_projection(
                [img_mean], self.MplWidget1.canvas.ax[0], plotMap=False, plotPoints=False)
            self.plot_projection(
                [img_diff], self.MplWidget1.canvas.ax[1], plotMap=False, plotPoints=False)
            self.plot_projection(
                [img_bicolor[(0, 1)]], self.MplWidget1.canvas.ax[2], plotMap=False, plotPoints=False)
            self.MplWidget1.canvas.draw()
            self.connect_measurement()
            
    @handle_exceptions(method_name="Changing projection region")
    def ChangeRegion(self):
        w = self.webmerc
        text = f'{w.lonmin},{w.lonmax},{w.latmin},{w.latmax},{w.pixel_per_km}'
        s = gui_string(
            text=text, caption='lonmin,lonmax,latmin,latmax,resolution_km')
        if s is not None:
            s = [float(x) for x in s.split(',')]
            if len(s) == 5:
                self.webmerc = ProjectionImageWebMercator(self.cloudimage, *s)
                print(self.webmerc)

    @handle_exceptions(method_name="Setting map region")
    def MapRegion(self):
        w = self.map_bounds
        text = f'{w[0]},{w[1]},{w[2]},{w[3]},{self.map_alpha}'
        s = gui_string(
            text=text, caption='lonmin,lonmax,latmin,latmax,map_alpha')
        if s is not None:
            s = [float(x) for x in s.split(',')]
            if len(s) == 5:
                self.map_bounds = s[:4]
                self.map_alpha = max(min(s[4], 1.0), 0.0)

    # onclick_digitize_control_points method now handled by ControlPointDigitizer

    @handle_exceptions(method_name="Control point digitization button click")
    def DigitizeControlPointsClick(self):
        if self.isDigitizeControlPoints is None:
            self.StartDigitizeControlPoints()
        else:
            self.StopDigitizeControlPoints()

    @handle_exceptions(method_name="Starting control point digitization")
    def StartDigitizeControlPoints(self):
        print('Starting control point digitization...')
        if self.isDigitizeControlPoints is None and self.cloudimage2 is not None:
            self.DrawImage(otrs=True, kontrolpunkti=False, plot_stars=False)
            print('Click on the first image to start digitizing control points.')
            # Get current axes from canvas
            ax1 = self.MplWidget1.canvas.ax[0]
            ax2 = self.MplWidget1.canvas.ax[1]
            
            # Create CloudImagePair if needed
            if self.cpair is None:
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
            
            # Create control point digitizer with current axes and cloudimage pair
            self.control_point_digitizer = ControlPointDigitizer(
                ax1, ax2, self.cpair, self, self.z1, self.z2)
            print('Control point digitization started. Click on the first image to select points.')
            self.control_point_digitizer.start_digitization()
            print('Control point digitization is active.')
            self.isDigitizeControlPoints = True  # Keep for compatibility

    @handle_exceptions(method_name="Stopping control point digitization")
    def StopDigitizeControlPoints(self):
        if self.isDigitizeControlPoints is not None and self.control_point_digitizer is not None:
            self.control_point_digitizer.stop_digitization()
            self.control_point_digitizer = None
        self.isDigitizeControlPoints = None
        self.update_ui_state()

    @handle_exceptions(method_name="Loading control points")
    def LoadControlPoints(self):
        if self.cloudimage2 is not None:
            matchfile = f'{os.path.split(self.cloudimage.filename)[0]}/{self.cloudimage.code}_{self.cloudimage2.code}.txt'
            # Use gui_fname function 
            matchfile = gui_fname(
                directory=matchfile,
                caption='Atbilstību fails',
                filter='(*.txt)')
            if matchfile != '':
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
                self.cpair.LoadCorrespondances(matchfile)
                self.DrawImage(otrs=True, kontrolpunkti=True)
        self.update_ui_state()

    @handle_exceptions(method_name="Saving height points to CSV")
    def SaveHeightPointsCSV(self):
        if self.cpair is not None:
            # Calculate height points
            llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            
            # Create default filename
            csvfile = os.path.splitext(self.cloudimage.filename)[0] + '_heights.csv'
            csvfile = gui_save_fname(
                directory=csvfile,
                caption='Augstumu punktu CSV fails',
                filter='(*.csv)')
            
            if csvfile != '':
                import pandas as pd
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Point_ID': range(1, len(llh[0]) + 1),
                    'Latitude': llh[0],
                    'Longitude': llh[1],
                    'Height_km': llh[2] / 1000.0,
                    'Height_m': llh[2],
                    'Ray_Distance_m': rayminimaldistance,
                    'Z_Error_m': z_intrinsic_error,
                    'Valid': valid
                })
                
                # Save to CSV
                df.to_csv(csvfile, index=False, float_format='%.6f')
                
                print(f'Height points saved to: {csvfile}')
                print(f'Total points: {len(llh[0])}, Valid points: {sum(valid)}')

    @handle_exceptions(method_name="Plotting match points")
    def plot_matches(self, ax, pairNo):
        if self.cpair is not None:
            for i, (x, y) in enumerate(self.cpair.correspondances[pairNo]):
                ax.plot(x, y, marker='o', fillstyle='none',
                        markeredgecolor='red')
                ax.annotate(str(i+1), xy=(x, y), xytext=(3, 3),
                            color='#AAFFAA', fontsize=16, textcoords='offset pixels')

    @handle_exceptions(method_name="Drawing control point heights")
    def DrawControlPointHeights(self):
        self.disconnect_measurement()
        
        if self.cpair is not None:
            self.MplWidget1.canvas.initplot([121, 122])
            ax = self.MplWidget1.canvas.ax[0]
            ax2 = self.MplWidget1.canvas.ax[1]

            z1, z2 = self.z1, self.z2

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

    @handle_exceptions(method_name="Creating height map")
    def CreateHeightMap(self):
        if self.cpair is not None:
            llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            valid[:] = True
            if any(valid):
                self.webmerc.cloudimage = self.cloudimage
                heightgrid = self.webmerc.PrepareHeightMap(
                    llh[1][valid], llh[0][valid], llh[2][valid])
                self.heightmap = HeightMap(self.webmerc)
                self.heightmap.heightmap = heightgrid
                self.heightmap.points = llh
                self.heightmap.validpoints = valid
                self.DrawHeightMap()
            else:
                print('Nevar izveidot augstumu karti - nav derīgu kontrolpunktu')
            self.update_ui_state()

    @handle_exceptions(method_name="Saving height map")
    def SaveHeightMap(self):
        if self.heightmap is not None:
            projfile = os.path.splitext(self.cloudimage.filename)[0]+'.hmp'
            projfile = gui_save_fname(
                directory=projfile,
                caption='Augstumu kartes fails',
                filter='(*.hmp)')
            if projfile != '':
                self.heightmap.save(projfile)
                print(f'Saved heightmap {projfile}')

    @handle_exceptions(method_name="Loading height map")
    def LoadHeightMap(self):
        if self.cloudimage is not None:
            projfile = os.path.splitext(self.cloudimage.filename)[0]+'.hmp'
            projfile = gui_fname(
                directory=projfile,
                caption='Augstumu kartes fails',
                filter='(*.hmp)')
            if projfile != '':
                self.heightmap = HeightMap.load(projfile)
                self.webmerc = self.heightmap.webmerc
                print(f'Loaded heightmap {projfile}')
            self.update_ui_state()

    @handle_exceptions(method_name="Drawing height map")
    def DrawHeightMap(self):
        self.disconnect_measurement()
        
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
            self.connect_measurement()
            self.MplWidget1.canvas.draw()

    @handle_exceptions()
    def CameraCalibrationParameters(self):
        """Open camera calibration parameters dialog"""
        accepted, new_params = show_camera_parameters_dialog(
            initial_params=self.app_settings.camera_calibration
        )
        
        if accepted:
            # Update the settings object
            self.app_settings.camera_calibration = new_params
            
            # Save to persist the changes
            self.app_settings.save_to_file()
            
            print('Kalibrēšanas parametri:', self.app_settings.camera_calibration.to_dict())
            print(f'Distortion: {new_params.get_distortion_description()}')
            print(f'Projection: {new_params.get_projection_description()}')

    @handle_exceptions(method_name="Camera modification")
    def CameraModification(self):
        """Open camera modification/creation parameters dialog"""
        if self.cloudimage is None:
            print('Nav ielādēts attēls - nepieciešams attēls kameras izveidei')
            return
            
        existing_camera = getattr(self.cloudimage, 'camera', None)
        
        accepted, modified_params = show_camera_modification_dialog(
            parent=self, 
            camera=existing_camera,
            cloudimage=self.cloudimage
        )
        
        if accepted:
            print('New camera parameters:')
            self.PrintCameraParameters()
            
            # Update UI state to reflect camera changes (enable/disable menu items)
            self.update_ui_state()
            
            # Refresh the display to show changes
            if hasattr(self, 'MplWidget1') and hasattr(self.MplWidget1, 'canvas'):
                if hasattr(self.MplWidget1.canvas, 'ax'):
                    self.DrawAltAzClick()
        else:
            action = 'izveide' if existing_camera is None else 'modificēšana'
            print(f'Kameras {action} atcelta')

    @handle_exceptions(method_name="Saving projected image")
    def SaveProjectedImage(self, jpg=True):
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
            projfile = gui_save_fname(
                directory=projfile,
                caption='Projicētais attēls',
                filter=f'(*{ext})')
            if projfile != '':
                jgwfile = os.path.splitext(projfile)[0]+extjgw
                wm = ProjectionImageWebMercator(None, 17, 33, 56, 63, 1.0)
                wm.__setstate__(self.projected_image[0])
                if jpg:
                    img = self.projected_image[1][:, :, 0:3]
                    # Save JPG with world file
                    wm.SaveJgw(jgwfile)
                    import skimage.io
                    skimage.io.imsave(projfile, img)
                    print('Fails saglabāts:', projfile)
                else:
                    img = self.projected_image[1]
                    # Save TIFF using rasterio with proper CRS
                    try:
                        wm.SaveGeoTiffRasterio(img, projfile)
                        print('GeoTIFF fails ar CRS informāciju saglabāts:', projfile)
                    except (ImportError, Exception) as e:
                        # Fallback to old method if rasterio is not available or fails
                        wm.SaveJgw(jgwfile)
                        import skimage.io
                        skimage.io.imsave(projfile, img)
                        if isinstance(e, ImportError):
                            print('TIFF fails ar world file saglabāts  (rasterio nav pieejams):', projfile)
                        else:
                            print(f'TIFF fails ar world file saglabāts (rasterio kļūda: {e}):', projfile)
                    
                    # Create KML overlay for TIFF files
                    from sudrabainiemakoni import savekml
                    savekml.mapOverlay(wm, img, self.projHeight, projfile, saveimage=False, cloudimage=self.cloudimage)

    @handle_exceptions(method_name="Adding star with Alt-Az coordinates")
    def addStarWithAltAz(self, name, x_coord, y_coord, az_deg, alt_deg):
        """
        Add a star reference with Alt-Az coordinates.
        This is useful when you know the Alt-Az coordinates of a reference point in the image.
        
        Args:
            name: Star name or identifier
            x_coord: X pixel coordinate
            y_coord: Y pixel coordinate  
            az_deg: Azimuth in degrees
            alt_deg: Altitude in degrees
        """
        if self.cloudimage is not None:
            star = self.cloudimage.addStarWithAltAz(name, [x_coord, y_coord], az_deg, alt_deg)
            print(f"Added star '{name}' with Alt-Az coordinates: {az_deg:.2f}°, {alt_deg:.2f}°")
            return star
        else:
            print("No cloud image loaded")
            return None


def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler to prevent GUI termination on exceptions."""
    import traceback
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("EXCEPTION CAUGHT:")
    print(tb)

if __name__ == '__main__':

    # Install the global exception handler
    sys.excepthook = excepthook
    
    app = QApplication(sys.argv)
    myapp = MainW()    
    myapp.show()
    sys.exit(app.exec_())