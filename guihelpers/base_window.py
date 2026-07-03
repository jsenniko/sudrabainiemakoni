import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui, QtCore
from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni.cloudimage_camera import CameraCalibrationParams
from sudrabainiemakoni.starreference import StarReference
from sudrabainiemakoni import cameraprojections, plots, plots_altazgrid

from guihelpers.camera_parameters import show_camera_parameters_dialog
from guihelpers.camera_modification import show_camera_modification_dialog
from guihelpers.settings import AppSettings
from guihelpers.qthelper import gui_fname, gui_save_fname, gui_string, file_dialog_manager
from guihelpers import smhelper
from guihelpers.exceptions import handle_exceptions
from guihelpers.star_digitizer import StarDigitizer
from guihelpers.catalog_star_overlay import CatalogStarOverlay
from guihelpers.catalog_settings_dialog import show_catalog_settings_dialog
from guihelpers.grid_settings_dialog import show_grid_settings_dialog, GridSettings


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


class BaseCloudImageWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.showMaximized()

        self.actionIelas_t_att_lu.triggered.connect(self.LoadImageClick)
        self.actionMainit_att_lu.triggered.connect(self.ChangeImageClick)
        self.actionKalibr_t_kameru.triggered.connect(self.CalibrateCameraClick)
        self.actionSaglab_t_projektu.triggered.connect(self.SaveProject)
        self.actionIelas_t_projektu.triggered.connect(self.LoadProject)
        self.actionHorizont_lo_koordin_tu_re_is.triggered.connect(self.DrawAltAzClick)
        self.actionAtt_lu.triggered.connect(self.DrawImageClick)
        self.actionCiparot_zvaigznes.triggered.connect(self.DigitizeStarsClick)
        self.actionIelas_t_kameru.triggered.connect(self.LoadCamera)
        self.actionSaglab_t_kameru.triggered.connect(self.SaveCamera)
        self.actionUzst_d_t_datumu.triggered.connect(self.SetDate)
        self.actionUzst_d_t_platumu_garumu_augstumu.triggered.connect(self.SetCoordinates)
        self.actionKameras_kalibr_cijas_parametri.triggered.connect(self.CameraCalibrationParameters)
        self.actionKameras_modifikacija.triggered.connect(self.CameraModification)

        self._setup_star_reference_menu()
        self._setup_auto_calibration_menu()
        self._setup_catalog_menu()
        self._setup_residual_menu()
        self._setup_star_display_menu()
        self._setup_grid_menu()

        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        self.cloudimage = None
        self.app_settings = AppSettings()
        self.load_settings_to_ui()

        self.isDigitizeStars = None
        self.star_digitizer = None

        self.catalog_overlay = None
        self.catalog_min_magnitude = -3
        self.catalog_max_magnitude = 4
        self.catalog_min_altitude = 0.0
        self.catalog_overshoot_px = 20
        self.show_catalog_stars = False

        self.residual_overlay = None
        self.show_residual_scatter = False
        self.show_residual_quiver = False

        self.show_star_circles = True
        self.show_star_names = True

        self.grid_settings = GridSettings()

    # -- Menu setup ----------------------------------------------------------

    def _setup_catalog_menu(self):
        from PyQt5.QtWidgets import QAction
        self.menuZ_m_t.addSeparator()

        self.actionShowCatalogStars = QAction("Show Catalog Stars", self)
        self.actionShowCatalogStars.setCheckable(True)
        self.actionShowCatalogStars.setChecked(False)
        self.actionShowCatalogStars.setEnabled(False)
        self.actionShowCatalogStars.triggered.connect(self.ToggleCatalogStars)
        self.menuZ_m_t.addAction(self.actionShowCatalogStars)

        self.actionCatalogSettings = QAction("Catalog Settings...", self)
        self.actionCatalogSettings.setEnabled(False)
        self.actionCatalogSettings.triggered.connect(self.ShowCatalogSettings)
        self.menuZ_m_t.addAction(self.actionCatalogSettings)

        self.actionTransferCatalogStars = QAction("Transfer Catalog Stars to Digitization", self)
        self.actionTransferCatalogStars.setEnabled(False)
        self.actionTransferCatalogStars.triggered.connect(self.TransferCatalogStars)
        self.menuZ_m_t.addAction(self.actionTransferCatalogStars)

    def _setup_residual_menu(self):
        from PyQt5.QtWidgets import QAction
        self.menuZ_m_t.addSeparator()

        self.actionShowResidualScatter = QAction("Show Residual Scatter", self)
        self.actionShowResidualScatter.setCheckable(True)
        self.actionShowResidualScatter.setChecked(False)
        self.actionShowResidualScatter.setEnabled(False)
        self.actionShowResidualScatter.triggered.connect(self.ToggleResidualScatter)
        self.menuZ_m_t.addAction(self.actionShowResidualScatter)

        self.actionShowResidualQuiver = QAction("Show Residual Vectors", self)
        self.actionShowResidualQuiver.setCheckable(True)
        self.actionShowResidualQuiver.setChecked(False)
        self.actionShowResidualQuiver.setEnabled(False)
        self.actionShowResidualQuiver.triggered.connect(self.ToggleResidualQuiver)
        self.menuZ_m_t.addAction(self.actionShowResidualQuiver)

    def _setup_star_display_menu(self):
        from PyQt5.QtWidgets import QAction
        self.menuZ_m_t.addSeparator()

        self.actionShowStarCircles = QAction("Show Star Circles", self)
        self.actionShowStarCircles.setCheckable(True)
        self.actionShowStarCircles.setChecked(True)
        self.actionShowStarCircles.triggered.connect(self.ToggleStarCircles)
        self.menuZ_m_t.addAction(self.actionShowStarCircles)

        self.actionShowStarNames = QAction("Show Star Names", self)
        self.actionShowStarNames.setCheckable(True)
        self.actionShowStarNames.setChecked(True)
        self.actionShowStarNames.triggered.connect(self.ToggleStarNames)
        self.menuZ_m_t.addAction(self.actionShowStarNames)

    def _setup_grid_menu(self):
        from PyQt5.QtWidgets import QAction, QMenu
        plots_menu = None
        for action in self.menubar.actions():
            if action.menu() and 'Att' in action.text():
                plots_menu = action.menu()
                break
        if plots_menu is None:
            plots_menu = QMenu("Grid", self)
            self.menubar.addMenu(plots_menu)

        plots_menu.addSeparator()

        self.actionGridSettings = QAction("Grid Settings...", self)
        self.actionGridSettings.triggered.connect(self.ShowGridSettings)
        plots_menu.addAction(self.actionGridSettings)

        self.actionExportGridOverlay = QAction("Export Image with Grid Overlay...", self)
        self.actionExportGridOverlay.setEnabled(False)
        self.actionExportGridOverlay.triggered.connect(self.ExportGridOverlay)
        plots_menu.addAction(self.actionExportGridOverlay)

    def _setup_star_reference_menu(self):
        from PyQt5.QtWidgets import QAction
        self.menuFails.addSeparator()

        self.actionSaveStarsAltAz = QAction("Save Stars as Alt-Az...", self)
        self.actionSaveStarsAltAz.setEnabled(False)
        self.actionSaveStarsAltAz.triggered.connect(self.SaveStarsAltAz)
        self.menuFails.addAction(self.actionSaveStarsAltAz)

        self.actionLoadStarsAltAz = QAction("Load Stars from Alt-Az...", self)
        self.actionLoadStarsAltAz.setEnabled(False)
        self.actionLoadStarsAltAz.triggered.connect(self.LoadStarsAltAz)
        self.menuFails.addAction(self.actionLoadStarsAltAz)

    def _setup_auto_calibration_menu(self):
        from PyQt5.QtWidgets import QAction
        self.menuZ_m_t.addSeparator()

        self.actionAutoStarMatch = QAction("Auto Star Matching...", self)
        self.actionAutoStarMatch.setEnabled(False)
        self.actionAutoStarMatch.triggered.connect(self.AutoStarMatch)
        self.menuZ_m_t.addAction(self.actionAutoStarMatch)

        self.actionAutoMatchSettings = QAction("Auto Star Matching Settings...", self)
        self.actionAutoMatchSettings.triggered.connect(self.ShowAutoMatchSettings)
        self.menuZ_m_t.addAction(self.actionAutoMatchSettings)

    # -- Settings ------------------------------------------------------------

    def load_settings_to_ui(self):
        if self.app_settings.last_directory:
            file_dialog_manager.last_directory = self.app_settings.last_directory

    def save_ui_to_settings(self):
        self.app_settings.last_directory = file_dialog_manager.last_directory
        self.app_settings.save_to_file()

    # -- UI state ------------------------------------------------------------

    def update_ui_state(self):
        has_camera = self.cloudimage is not None and hasattr(self.cloudimage, 'camera') and self.cloudimage.camera is not None

        self.actionMainit_att_lu.setEnabled(self.cloudimage is not None)
        self.actionKalibr_t_kameru.setEnabled(self.cloudimage is not None)
        self.actionSaglab_t_projektu.setEnabled(self.cloudimage is not None)
        self.actionHorizont_lo_koordin_tu_re_is.setEnabled(has_camera)
        self.actionAtt_lu.setEnabled(self.cloudimage is not None)
        self.actionCiparot_zvaigznes.setEnabled(self.cloudimage is not None)
        self.actionIelas_t_kameru.setEnabled(self.cloudimage is not None)
        self.actionSaglab_t_kameru.setEnabled(has_camera)
        self.actionKameras_modifikacija.setEnabled(self.cloudimage is not None)
        self.actionUzst_d_t_datumu.setEnabled(self.cloudimage is not None)

        has_stars = self.cloudimage is not None and len(self.cloudimage.starReferences) > 0
        has_calibration = has_camera and has_stars

        self.actionShowCatalogStars.setEnabled(has_camera)
        self.actionCatalogSettings.setEnabled(has_camera)
        self.actionTransferCatalogStars.setEnabled(has_camera and self.show_catalog_stars)
        self.actionShowResidualScatter.setEnabled(has_calibration)
        self.actionShowResidualQuiver.setEnabled(has_calibration)
        self.actionSaveStarsAltAz.setEnabled(has_camera and has_stars)
        self.actionLoadStarsAltAz.setEnabled(has_camera)
        self.actionExportGridOverlay.setEnabled(has_camera)
        self.actionAutoStarMatch.setEnabled(has_camera)

    # -- Qt events -----------------------------------------------------------

    def onUpdateText(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.save_ui_to_settings()
        super().closeEvent(event)

    # -- Image load/change ---------------------------------------------------

    def LoadImageClick(self):
        filename_jpg = gui_fname(caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg != '':
            self.LoadImage(filename_jpg)

    @handle_exceptions(method_name="Loading image")
    def LoadImage(self, filename_jpg):
        self.console.clear()
        case_id = os.path.splitext(os.path.split(filename_jpg)[1])[0]
        filename_stars = smhelper.check_stars_file(filename_jpg)
        lat, lon, height = smhelper.check_latlon_file(filename_jpg)
        self.cloudimage = CloudImage.from_files(
            case_id, filename_jpg, filename_stars, lat, lon, height=height)
        self.isDigitizeStars = None
        self.DrawImage()
        self.update_ui_state()

    def ChangeImageClick(self):
        filename_jpg = gui_fname(caption="Sudrabaino mākoņu attēls", filter='*.jpg')
        if filename_jpg != '':
            self.ChangeImage(filename_jpg)

    @handle_exceptions(method_name="Changing image")
    def ChangeImage(self, filename_jpg):
        if self.cloudimage is not None:
            self.cloudimage.filename = filename_jpg
            self.cloudimage.LoadImage(reload=True)
        self.update_ui_state()

    # -- Date / coordinates --------------------------------------------------

    @handle_exceptions(method_name="Setting date")
    def SetDate(self):
        d = self.cloudimage.date.to_datetime(timezone=CloudImage.timezone)
        s = d.strftime('%Y-%m-%dT%H:%M:%S')
        s = gui_string(text=s, caption='Ievadi datumu, YYYY-MM-DDTHH:MM:SS')
        if s is not None:
            import datetime
            d = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            self.cloudimage.setDate(d)

    @handle_exceptions(method_name="Setting coordinates")
    def SetCoordinates(self):
        lat, lon, height = self.cloudimage.getLocation()
        s = f'{lat:.5f},{lon:.5f},{height:.0f}'
        s = gui_string(text=s, caption='Ievadi koordinātes (lat,lon,z)')
        if s is not None:
            s = s.split(',')
            lat, lon, height = float(s[0]), float(s[1]), 0.0
            if len(s) > 2:
                height = float(s[2])
            self.cloudimage.setLocation(lon=lon, lat=lat, height=height)

    # -- Camera --------------------------------------------------------------

    def PrintCameraParameters(self):
        if self.cloudimage is not None:
            cldim = self.cloudimage
            az, el, rot = cldim.camera.get_azimuth_elevation_rotation()
            print(f'Kameras ass azimuts {az:.2f}°')
            print(f'Kameras ass augstums virs horizonta {el:.2f}°')
            print(f'Kameras pagrieziena leņķis {rot:.2f}°')
            fx, fy, cx, cy = cldim.camera.get_focal_lengths_mm()
            print(f'Kameras fokusa attālumi (35mm ekvivalents) {fx:.1f} {fy:.1f}')
            print(f'Kameras ass pozīcija {cx:.1f} {cy:.1f}')
            k1 = cldim.camera.camera_enu.k1
            k2 = cldim.camera.camera_enu.k2
            k3 = cldim.camera.camera_enu.k3
            distortion_str = f'Distortion: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}'
            p1 = getattr(cldim.camera.camera_enu, 'p1', None)
            p2 = getattr(cldim.camera.camera_enu, 'p2', None)
            if p1 is not None and p2 is not None:
                distortion_str += f', p1={p1:.6f}, p2={p2:.6f}'
            k4 = getattr(cldim.camera.camera_enu, 'k4', None)
            k5 = getattr(cldim.camera.camera_enu, 'k5', None)
            k6 = getattr(cldim.camera.camera_enu, 'k6', None)
            if k4 is not None and k5 is not None and k6 is not None:
                distortion_str += f', k4={k4:.6f}, k5={k5:.6f}, k6={k6:.6f}'
            print(distortion_str)
            print(f'Projection: {cameraprojections.name_by_projection(cldim.camera.camera_enu.projection)}')

    @handle_exceptions(method_name="Camera calibration")
    def CalibrateCameraClick(self):
        if self.cloudimage is not None:
            self.cloudimage.PrepareCamera(**self.app_settings.camera_calibration.to_dict())
            self.PrintCameraParameters()
            self.DrawAltAzClick()
        self.update_ui_state()

    @handle_exceptions(method_name="Loading camera")
    def LoadCamera(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0] + '_enu.json'
        camfile = gui_fname(directory=camfile, caption='Kameras fails', filter='(*.json)')
        if camfile != '':
            self.cloudimage.LoadCamera(camfile)

    @handle_exceptions(method_name="Saving camera")
    def SaveCamera(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0] + '_enu.json'
        camfile = gui_save_fname(directory=camfile, caption='Kameras fails', filter='(*.json)')
        if camfile != '':
            self.cloudimage.SaveCamera(camfile)

    @handle_exceptions()
    def CameraCalibrationParameters(self):
        accepted, new_params = show_camera_parameters_dialog(
            initial_params=self.app_settings.camera_calibration)
        if accepted:
            self.app_settings.camera_calibration = new_params
            self.app_settings.save_to_file()
            print('Kalibrēšanas parametri:', self.app_settings.camera_calibration.to_dict())
            print(f'Distortion: {new_params.get_distortion_description()}')
            print(f'Projection: {new_params.get_projection_description()}')

    @handle_exceptions(method_name="Camera modification")
    def CameraModification(self):
        if self.cloudimage is None:
            print('Nav ielādēts attēls - nepieciešams attēls kameras izveidei')
            return
        existing_camera = getattr(self.cloudimage, 'camera', None)
        print('existing_camera:', existing_camera)
        print('existing_camera.camera_enu:', existing_camera.camera_enu if existing_camera is not None else None)
        if existing_camera is not None and existing_camera.camera_enu is None:
            existing_camera = None
        print('existing_camera passed to dialog:', existing_camera)
        accepted, modified_params = show_camera_modification_dialog(
            parent=self, camera=existing_camera, cloudimage=self.cloudimage)
        if accepted:
            print('camera_enu after dialog:', self.cloudimage.camera.camera_enu)
            print('New camera parameters:')
            self.PrintCameraParameters()
            self.update_ui_state()
            if hasattr(self, 'MplWidget1') and hasattr(self.MplWidget1, 'canvas'):
                if hasattr(self.MplWidget1.canvas, 'ax'):
                    self.DrawAltAzClick()
        else:
            action = 'izveide' if existing_camera is None else 'modificēšana'
            print(f'Kameras {action} atcelta')

    # -- Project load/save ---------------------------------------------------

    @handle_exceptions(method_name="Loading project")
    def LoadProject(self):
        projfile = gui_fname(caption='Projekta fails', filter='(*.proj)')
        if projfile != '':
            print(f'Loading project {projfile}')
            self.console.clear()
            self.cloudimage = CloudImage.load(projfile)
            self.isDigitizeStars = None
            print(f'Loaded project file {projfile}')
            print(self.cloudimage)
            self.DrawImage()
        self.update_ui_state()

    @handle_exceptions(method_name="Saving project")
    def SaveProject(self):
        projfile = os.path.splitext(self.cloudimage.filename)[0] + '.proj'
        projfile = gui_save_fname(directory=projfile, caption='Projekta fails', filter='(*.proj)')
        if projfile != '':
            self.cloudimage.save(projfile)
            print(f'Saved project file {projfile}')

    # -- Drawing -------------------------------------------------------------

    @handle_exceptions(method_name="Drawing Alt-Az grid")
    def DrawAltAzClick(self):
        if self.cloudimage is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            grid_kwargs = self.grid_settings.to_grid_kwargs()
            plots_altazgrid.PlotAltAzGrid_v2(
                self.cloudimage.imagearray, self.cloudimage.camera.camera_enu,
                ax=ax, grid_kwargs=grid_kwargs)

            image_height, image_width = self.cloudimage.imagearray.shape[:2]
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)

            cldim = self.cloudimage
            enu_unit_coords = cldim.get_stars_enu_unit_coords()
            if len(enu_unit_coords) > 0:
                campx = cldim.camera.camera_enu.imageFromSpace(enu_unit_coords)
                for sr, cpx in zip(cldim.starReferences, campx):
                    ix, iy = sr.pixelcoords
                    if self.show_star_circles:
                        p = ax.plot(ix, iy, marker='o', fillstyle='none')
                    else:
                        p = ax.plot(ix, iy, marker='o', fillstyle='none', alpha=0)
                    if self.show_star_names:
                        ax.annotate(sr.name, xy=(ix, iy), xytext=(3, 3),
                                    color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                    ax.plot(cpx[0], cpx[1], marker='x', fillstyle='none', color=p[0].get_color())

            if self.show_catalog_stars:
                self._ensure_catalog_overlay()
                if self.catalog_overlay is not None:
                    self.catalog_overlay.update_catalog()
                    self.catalog_overlay.draw()

            if self.show_residual_scatter or self.show_residual_quiver:
                if self._update_residual_overlay():
                    if self.show_residual_scatter:
                        self.residual_overlay.show_scatter()
                    if self.show_residual_quiver:
                        self.residual_overlay.show_quiver()

            self.MplWidget1.canvas.draw()

    @handle_exceptions(method_name="Drawing image")
    def DrawImageClick(self):
        self.DrawImage()

    @handle_exceptions(method_name="Drawing image")
    def DrawImage(self, plot_stars=True):
        if self.cloudimage is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            ax.imshow(self.cloudimage.imagearray)
            if plot_stars:
                plots.PlotStars(self.cloudimage, ax,
                                show_circles=self.show_star_circles,
                                show_names=self.show_star_names)

            if self.show_catalog_stars:
                self._ensure_catalog_overlay()
                if self.catalog_overlay is not None:
                    self.catalog_overlay.update_catalog()
                    self.catalog_overlay.draw()

            if self.show_residual_scatter or self.show_residual_quiver:
                if self._update_residual_overlay():
                    if self.show_residual_scatter:
                        self.residual_overlay.show_scatter()
                    if self.show_residual_quiver:
                        self.residual_overlay.show_quiver()

            self.MplWidget1.canvas.draw()
            return ax

    # -- Star digitization ---------------------------------------------------

    @handle_exceptions(method_name="Starting star digitization")
    def StartDigitizeStars(self):
        if self.isDigitizeStars is None:
            self.DrawImage(plot_stars=False)
            ax = self.MplWidget1.canvas.ax
            if hasattr(ax, '__len__'):
                ax = ax[0]

            self._ensure_catalog_overlay()
            if self.catalog_overlay is not None and self.catalog_overlay.catalog_df is None:
                self.catalog_overlay.update_catalog()

            if self.cloudimage.camera is not None and len(self.cloudimage.starReferences) > 0:
                self._ensure_residual_overlay()

            self.star_digitizer = StarDigitizer(
                ax, self.cloudimage, self,
                catalog_overlay=self.catalog_overlay,
                residual_overlay=self.residual_overlay)
            self.star_digitizer.start_digitization()
            self.isDigitizeStars = True

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

    # -- Catalog overlay -----------------------------------------------------

    def _ensure_catalog_overlay(self):
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            return False
        ax = self.MplWidget1.canvas.ax
        if hasattr(ax, '__len__'):
            ax = ax[0]
        if self.catalog_overlay is None:
            self.catalog_overlay = CatalogStarOverlay(
                ax, self.cloudimage.camera, self.cloudimage.location, self.cloudimage.date,
                min_magnitude=self.catalog_min_magnitude,
                max_magnitude=self.catalog_max_magnitude,
                overshoot_px=self.catalog_overshoot_px,
                min_altitude=self.catalog_min_altitude)
        else:
            self.catalog_overlay.ax = ax
            self.catalog_overlay.camera = self.cloudimage.camera
            self.catalog_overlay.location = self.cloudimage.location
            self.catalog_overlay.observation_time = self.cloudimage.date
            self.catalog_overlay.set_magnitude_range(self.catalog_min_magnitude, self.catalog_max_magnitude)
            self.catalog_overlay.set_altitude_filter(self.catalog_min_altitude)
        return True

    @handle_exceptions(method_name="Toggling catalog stars")
    def ToggleCatalogStars(self):
        self.show_catalog_stars = self.actionShowCatalogStars.isChecked()
        if not self._ensure_catalog_overlay():
            self.actionShowCatalogStars.setChecked(False)
            return
        if self.show_catalog_stars:
            print("Updating catalog...")
            self.catalog_overlay.update_catalog()
            self.catalog_overlay.show()
        else:
            self.catalog_overlay.hide()
        self.update_ui_state()

    @handle_exceptions(method_name="Showing catalog settings")
    def ShowCatalogSettings(self):
        if self.cloudimage is None:
            return
        accepted, (min_mag, max_mag, min_alt, overshoot) = show_catalog_settings_dialog(
            parent=self,
            min_magnitude=self.catalog_min_magnitude,
            max_magnitude=self.catalog_max_magnitude,
            min_altitude=self.catalog_min_altitude,
            overshoot_px=self.catalog_overshoot_px)
        if accepted:
            self.catalog_min_magnitude = min_mag
            self.catalog_max_magnitude = max_mag
            self.catalog_min_altitude = min_alt
            self.catalog_overshoot_px = overshoot
            print(f"Catalog settings: mag {min_mag} to {max_mag}, alt >= {min_alt}°, overshoot {overshoot}px")
            if self.show_catalog_stars and self.catalog_overlay is not None:
                self.catalog_overlay.set_magnitude_range(min_mag, max_mag)
                self.catalog_overlay.set_altitude_filter(min_alt)
                self.catalog_overlay.overshoot_px = overshoot
                self.catalog_overlay.refresh()

    @handle_exceptions(method_name="Transferring catalog stars")
    def TransferCatalogStars(self):
        if not self._ensure_catalog_overlay():
            return
        if self.catalog_overlay.catalog_df is None or len(self.catalog_overlay.catalog_df) == 0:
            print("No catalog stars available")
            return
        stars_to_transfer = self.catalog_overlay.get_all_visible_stars()
        if stars_to_transfer is None or len(stars_to_transfer) == 0:
            print("No catalog stars to transfer")
            return
        from guihelpers.qthelper import gui_confirm
        if not gui_confirm(caption=f"Transfer {len(stars_to_transfer)} catalog stars to digitization?"):
            return
        transferred_count = 0
        for _, star in stars_to_transfer.iterrows():
            star_exists = any(sr.name == star['name'] for sr in self.cloudimage.starReferences)
            if not star_exists:
                star_ref = StarReference(
                    f"ra:{star['ra']:.6f},{star['dec']:.6f}",
                    [star['pixel_x'], star['pixel_y']])
                star_ref.name = star['name']
                self.cloudimage.starReferences.append(star_ref)
                transferred_count += 1
        print(f"Transferred {transferred_count} catalog stars to digitization")
        print(f"Total digitized stars: {len(self.cloudimage.starReferences)}")
        if self.isDigitizeStars and self.star_digitizer is not None:
            self.star_digitizer._load_existing_stars()
        else:
            self.DrawImage()

    # -- Residual overlay ----------------------------------------------------

    def _ensure_residual_overlay(self):
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("Cannot create residual overlay: no camera available")
            return False
        if len(self.cloudimage.starReferences) == 0:
            print("Cannot create residual overlay: no star references")
            return False
        ax = self.MplWidget1.canvas.ax
        if hasattr(ax, '__len__'):
            ax = ax[0]
        if self.residual_overlay is None:
            from guihelpers.residual_overlay import ResidualOverlay
            self.residual_overlay = ResidualOverlay(ax)
        else:
            self.residual_overlay.ax = ax
        return True

    def _update_residual_overlay(self):
        if not self._ensure_residual_overlay():
            return False
        residual_data = self.cloudimage.camera.calculate_residuals(
            self.cloudimage.starReferences,
            self.cloudimage.location,
            self.cloudimage.date)
        if residual_data is None:
            print("Failed to calculate residuals")
            return False
        self.residual_overlay.set_residuals(
            residual_data['star_pixel_coords'],
            residual_data['model_pixel_coords'])
        print(f"Residuals updated: RMS = {residual_data['rms']:.2f} pixels")
        return True

    @handle_exceptions(method_name="Toggling residual scatter")
    def ToggleResidualScatter(self):
        self.show_residual_scatter = self.actionShowResidualScatter.isChecked()
        if not self._update_residual_overlay():
            self.actionShowResidualScatter.setChecked(False)
            return
        if self.show_residual_scatter:
            self.residual_overlay.show_scatter()
        else:
            self.residual_overlay.hide_scatter()
        self.MplWidget1.canvas.draw()

    @handle_exceptions(method_name="Toggling residual quiver")
    def ToggleResidualQuiver(self):
        self.show_residual_quiver = self.actionShowResidualQuiver.isChecked()
        if not self._update_residual_overlay():
            self.actionShowResidualQuiver.setChecked(False)
            return
        if self.show_residual_quiver:
            self.residual_overlay.show_quiver()
        else:
            self.residual_overlay.hide_quiver()
        self.MplWidget1.canvas.draw()

    # -- Star display toggles ------------------------------------------------

    @handle_exceptions(method_name="Toggling star circles")
    def ToggleStarCircles(self):
        self.show_star_circles = self.actionShowStarCircles.isChecked()
        self.DrawImage()

    @handle_exceptions(method_name="Toggling star names")
    def ToggleStarNames(self):
        self.show_star_names = self.actionShowStarNames.isChecked()
        self.DrawImage()

    # -- Alt-Az file I/O -----------------------------------------------------

    @handle_exceptions(method_name="Saving stars as Alt-Az")
    def SaveStarsAltAz(self):
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available to convert stars to Alt-Az")
            return
        if len(self.cloudimage.starReferences) == 0:
            print("No stars to save")
            return
        default_filename = os.path.splitext(self.cloudimage.filename)[0] + '_altaz.txt'
        filename = gui_save_fname(directory=default_filename, caption='Save stars as Alt-Az', filter='(*.txt)')
        if filename == '':
            return
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Star references in Alt-Az format\n")
                f.write("# Name\tPixel_X\tPixel_Y\tAzimuth\tAltitude\n")
                for star_ref in self.cloudimage.starReferences:
                    px, py = star_ref.pixelcoords
                    if star_ref.hasDirectAltAz():
                        az = star_ref.altaz_coord.az.deg
                        alt = star_ref.altaz_coord.alt.deg
                    else:
                        altaz = star_ref.getAltAzCoord(self.cloudimage.altaz)
                        az = altaz.az.deg
                        alt = altaz.alt.deg
                    f.write(f"{star_ref.name}\t{px:.2f}\t{py:.2f}\t{az:.6f}\t{alt:.6f}\n")
            print(f"Saved {len(self.cloudimage.starReferences)} stars to {filename}")
        except Exception as e:
            print(f"Error saving stars: {e}")

    @handle_exceptions(method_name="Loading stars from Alt-Az")
    def LoadStarsAltAz(self):
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available - cannot load Alt-Az stars")
            return
        filename = gui_fname(caption='Load stars from Alt-Az file', filter='(*.txt)')
        if filename == '':
            return
        try:
            loaded_count = 0
            skipped_count = 0
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 5:
                        parts = line.split()
                    if len(parts) < 5:
                        print(f"Skipping invalid line: {line}")
                        skipped_count += 1
                        continue
                    try:
                        name = parts[0]
                        px = float(parts[1])
                        py = float(parts[2])
                        az = float(parts[3])
                        alt = float(parts[4])
                        star_ref = StarReference(f"{az:.6f},{alt:.6f}", [px, py])
                        star_ref.name = f'AZ:{name}'
                        self.cloudimage.starReferences.append(star_ref)
                        loaded_count += 1
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line '{line}': {e}")
                        skipped_count += 1
            print(f"Loaded {loaded_count} stars from {filename}")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} stars (duplicates or invalid)")
            print(f"Total stars: {len(self.cloudimage.starReferences)}")
            if self.isDigitizeStars and self.star_digitizer is not None:
                self.star_digitizer._load_existing_stars()
            else:
                self.DrawImage()
        except Exception as e:
            print(f"Error loading stars: {e}")

    # -- Grid ----------------------------------------------------------------

    @handle_exceptions(method_name="Showing grid settings")
    def ShowGridSettings(self):
        accepted, new_settings = show_grid_settings_dialog(parent=self, settings=self.grid_settings)
        if accepted:
            self.grid_settings = new_settings
            print("Grid settings updated")
            if hasattr(self, 'MplWidget1') and hasattr(self.MplWidget1, 'canvas'):
                if hasattr(self.MplWidget1.canvas, 'ax'):
                    self.DrawAltAzClick()

    @handle_exceptions(method_name="Exporting grid overlay")
    def ExportGridOverlay(self):
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available to export grid")
            return
        default_filename = os.path.splitext(self.cloudimage.filename)[0] + '_grid.jpg'
        filename = gui_save_fname(directory=default_filename,
                                  caption='Export Image with Grid Overlay', filter='(*.jpg *.png)')
        if filename == '':
            return
        try:
            grid_kwargs = self.grid_settings.to_grid_kwargs()
            plots.CreateGridOverlayFigure(
                self.cloudimage, grid_kwargs=grid_kwargs, dpi=100,
                filename=filename, close_figure=True)
            h_px, w_px = self.cloudimage.imagearray.shape[:2]
            print(f"Exported grid overlay to {filename}")
            print(f"Image size: {w_px}x{h_px} pixels")
        except Exception as e:
            print(f"Error exporting grid overlay: {e}")
            import traceback
            traceback.print_exc()

    # -- Auto star matching --------------------------------------------------

    @handle_exceptions(method_name="Auto star matching")
    def AutoStarMatch(self):
        if self.cloudimage is None or self.cloudimage.camera is None:
            print("No camera loaded - cannot run auto star matching")
            return

        star_extract_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'sudrabainiemakoni', 'calibration', 'StarExtract'))
        if star_extract_dir not in sys.path:
            sys.path.insert(0, star_extract_dir)

        from grid_star_detection import detect_stars_grid
        from star_matching import calibrate_camera_pose, StarMatchError
        import pandas as pd

        print("Detecting stars in image...")
        detected_stars = pd.DataFrame(detect_stars_grid(self.cloudimage.imagearray))
        detected_stars = detected_stars.sort_values('flux', ascending=False).reset_index(drop=True)
        detected_xy = detected_stars[['x', 'y']].values
        print(f"  Detected {len(detected_xy)} stars")

        s = self.app_settings.auto_match
        print(f"Running automatic star matching pipeline (mag_coarse={s.max_magnitude_coarse}, "
              f"angle_tol={s.angle_tol_deg}deg, fl_unc={s.focal_length_uncertainty}, "
              f"n_search={s.n_search}, mag_fine={s.max_magnitude_fine}, "
              f"nn_dist={s.nn_max_dist_px}px, optimize_intrinsics={s.optimize_intrinsics})...")
        try:
            refined_camera_enu, results = calibrate_camera_pose(
                detected_xy,
                self.cloudimage.camera.camera_enu,
                self.cloudimage.location,
                self.cloudimage.date,
                max_magnitude_fine=5.5,
                angle_tol_deg=0.1,
                debug_star='Leo 60')
        except StarMatchError as e:
            print(f"Auto star matching failed: {e}")
            return

        rms = results['rms_px']
        n_stars = len(results['detected_xy'])
        print(f"Calibration succeeded: {n_stars} stars, RMS = {rms:.2f} px")

        if s.update_camera:
            from sudrabainiemakoni.cloudimage_camera import Camera
            from sudrabainiemakoni import cameraprojections
            from sudrabainiemakoni.cloudimage_camera import name_by_distortion
            old_enu = self.cloudimage.camera.camera_enu
            image_size = self.cloudimage.camera.image_size
            proj_name = cameraprojections.name_by_projection(old_enu.projection)
            dist_name = name_by_distortion(old_enu.lens)
            new_camera = Camera.from_manual_parameters(
                image_size=image_size,
                location=self.cloudimage.location,
                fx=refined_camera_enu.focallength_x_px,
                fy=refined_camera_enu.focallength_y_px,
                cx=refined_camera_enu.center_x_px,
                cy=refined_camera_enu.center_y_px,
                azimuth=refined_camera_enu.heading_deg,
                elevation=refined_camera_enu.tilt_deg - 90,
                rotation=refined_camera_enu.roll_deg,
                k1=old_enu.k1, k2=old_enu.k2, k3=old_enu.k3,
                p1=getattr(old_enu, 'p1', 0), p2=getattr(old_enu, 'p2', 0),
                k4=getattr(old_enu, 'k4', 0), k5=getattr(old_enu, 'k5', 0),
                k6=getattr(old_enu, 'k6', 0),
                projection=proj_name, distortion_type=dist_name)
            self.cloudimage.camera = new_camera
            print("Camera updated")
        else:
            print("Camera not updated (update_camera=False)")

        self.cloudimage.starReferences = []
        catalog_rows = results['catalog_rows']
        matched_det_xy = results['detected_xy']
        for i in range(len(matched_det_xy)):
            px, py = matched_det_xy[i]
            row = catalog_rows.iloc[i]
            star = StarReference(
                f"ra:{float(row['ra']):.6f},{float(row['dec']):.6f}",
                [float(px), float(py)])
            star.name = str(row['name'])
            self.cloudimage.starReferences.append(star)

        print(f"Added {len(matched_det_xy)} star references")
        self.update_ui_state()
        self.DrawAltAzClick()

    @handle_exceptions(method_name="Auto match settings")
    def ShowAutoMatchSettings(self):
        from guihelpers.auto_match_settings_dialog import show_auto_match_settings_dialog
        accepted, new_settings = show_auto_match_settings_dialog(
            parent=self, settings=self.app_settings.auto_match)
        if accepted:
            self.app_settings.auto_match = new_settings
            self.app_settings.save_to_file()
            print("Auto match settings saved")

    # -- Misc ----------------------------------------------------------------

    @handle_exceptions(method_name="Adding star with Alt-Az coordinates")
    def addStarWithAltAz(self, name, x_coord, y_coord, az_deg, alt_deg):
        if self.cloudimage is not None:
            star = self.cloudimage.addStarWithAltAz(name, [x_coord, y_coord], az_deg, alt_deg)
            print(f"Added star '{name}' with Alt-Az coordinates: {az_deg:.2f}°, {alt_deg:.2f}°")
            return star
        else:
            print("No cloud image loaded")
            return None


def excepthook(exc_type, exc_value, exc_tb):
    import traceback
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("EXCEPTION CAUGHT:")
    print(tb)
