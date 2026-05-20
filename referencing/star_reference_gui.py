import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui, QtCore
from sudrabainiemakoni.cloudimage import CloudImage
from sudrabainiemakoni.cloudimage_camera import CameraCalibrationParams
from sudrabainiemakoni.starreference import StarReference
from sudrabainiemakoni import cameraprojections, plots, plots_altazgrid

# Import from guihelpers (separate top-level package)
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

# Import local UI file
try:
    from .star_reference_ui import Ui_MainWindow
except ImportError:
    from star_reference_ui import Ui_MainWindow


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


class StarReferenceWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.showMaximized()

        # Connect only relevant menu actions
        self.actionIelas_t_att_lu.triggered.connect(self.LoadImageClick)
        self.actionMainit_att_lu.triggered.connect(self.ChangeImageClick)
        self.actionKalibr_t_kameru.triggered.connect(self.CalibrateCameraClick)
        self.actionSaglab_t_projektu.triggered.connect(self.SaveProject)
        self.actionIelas_t_projektu.triggered.connect(self.LoadProject)
        self.actionHorizont_lo_koordin_tu_re_is.triggered.connect(
            self.DrawAltAzClick)
        self.actionAtt_lu.triggered.connect(self.DrawImageClick)
        self.actionCiparot_zvaigznes.triggered.connect(
            self.DigitizeStarsClick)
        self.actionIelas_t_kameru.triggered.connect(self.LoadCamera)
        self.actionSaglab_t_kameru.triggered.connect(self.SaveCamera)
        self.actionUzst_d_t_datumu.triggered.connect(self.SetDate)
        self.actionUzst_d_t_platumu_garumu_augstumu.triggered.connect(self.SetCoordinates)
        self.actionKameras_kalibr_cijas_parametri.triggered.connect(
            self.CameraCalibrationParameters)
        self.actionKameras_modifikacija.triggered.connect(
            self.CameraModification)

        # Add star reference save/load actions
        self._setup_star_reference_menu()

        # Add auto star matching action
        self._setup_auto_calibration_menu()

        # Redirect console output
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        self.cloudimage = None
        self.app_settings = AppSettings()
        self.load_settings_to_ui()

        self.isDigitizeStars = None
        self.star_digitizer = None

        # Catalog star overlay
        self.catalog_overlay = None
        self.catalog_min_magnitude = -3
        self.catalog_max_magnitude = 4
        self.catalog_min_altitude = 0.0
        self.catalog_overshoot_px = 20
        self.show_catalog_stars = False

        # Residual overlay
        self.residual_overlay = None
        self.show_residual_scatter = False
        self.show_residual_quiver = False

        # Star display toggles
        self.show_star_circles = True
        self.show_star_names = True

        # Grid settings
        self.grid_settings = GridSettings()

        # Add catalog menu actions dynamically
        self._setup_catalog_menu()
        self._setup_residual_menu()
        self._setup_star_display_menu()
        self._setup_grid_menu()

    def _setup_catalog_menu(self):
        """Setup catalog star menu actions."""
        from PyQt5.QtWidgets import QAction

        # Add catalog submenu to the Stars menu
        self.menuZ_m_t.addSeparator()

        # Show/hide catalog stars action (checkbox)
        self.actionShowCatalogStars = QAction("Show Catalog Stars", self)
        self.actionShowCatalogStars.setCheckable(True)
        self.actionShowCatalogStars.setChecked(False)
        self.actionShowCatalogStars.setEnabled(False)
        self.actionShowCatalogStars.triggered.connect(self.ToggleCatalogStars)
        self.menuZ_m_t.addAction(self.actionShowCatalogStars)

        # Catalog settings action
        self.actionCatalogSettings = QAction("Catalog Settings...", self)
        self.actionCatalogSettings.setEnabled(False)
        self.actionCatalogSettings.triggered.connect(self.ShowCatalogSettings)
        self.menuZ_m_t.addAction(self.actionCatalogSettings)

        # Transfer stars action
        self.actionTransferCatalogStars = QAction("Transfer Catalog Stars to Digitization", self)
        self.actionTransferCatalogStars.setEnabled(False)
        self.actionTransferCatalogStars.triggered.connect(self.TransferCatalogStars)
        self.menuZ_m_t.addAction(self.actionTransferCatalogStars)

    def _setup_residual_menu(self):
        """Setup residual visualization menu actions."""
        from PyQt5.QtWidgets import QAction

        # Add residual submenu to the Stars menu
        self.menuZ_m_t.addSeparator()

        # Show/hide residual scatter action (checkbox)
        self.actionShowResidualScatter = QAction("Show Residual Scatter", self)
        self.actionShowResidualScatter.setCheckable(True)
        self.actionShowResidualScatter.setChecked(False)
        self.actionShowResidualScatter.setEnabled(False)
        self.actionShowResidualScatter.triggered.connect(self.ToggleResidualScatter)
        self.menuZ_m_t.addAction(self.actionShowResidualScatter)

        # Show/hide residual quiver action (checkbox)
        self.actionShowResidualQuiver = QAction("Show Residual Vectors", self)
        self.actionShowResidualQuiver.setCheckable(True)
        self.actionShowResidualQuiver.setChecked(False)
        self.actionShowResidualQuiver.setEnabled(False)
        self.actionShowResidualQuiver.triggered.connect(self.ToggleResidualQuiver)
        self.menuZ_m_t.addAction(self.actionShowResidualQuiver)

    def _setup_star_display_menu(self):
        """Setup star display toggle menu actions."""
        from PyQt5.QtWidgets import QAction

        # Add separator
        self.menuZ_m_t.addSeparator()

        # Show/hide star circles action (checkbox)
        self.actionShowStarCircles = QAction("Show Star Circles", self)
        self.actionShowStarCircles.setCheckable(True)
        self.actionShowStarCircles.setChecked(True)  # Default on
        self.actionShowStarCircles.triggered.connect(self.ToggleStarCircles)
        self.menuZ_m_t.addAction(self.actionShowStarCircles)

        # Show/hide star names action (checkbox)
        self.actionShowStarNames = QAction("Show Star Names", self)
        self.actionShowStarNames.setCheckable(True)
        self.actionShowStarNames.setChecked(True)  # Default on
        self.actionShowStarNames.triggered.connect(self.ToggleStarNames)
        self.menuZ_m_t.addAction(self.actionShowStarNames)

    def _setup_grid_menu(self):
        """Setup grid menu actions."""
        from PyQt5.QtWidgets import QAction

        # Add grid settings action to Plots menu (assuming it exists, or add to File menu)
        # Find or create Plots menu
        plots_menu = None
        for action in self.menubar.actions():
            if action.menu() and 'Att' in action.text():  # Finding the image menu
                plots_menu = action.menu()
                break

        if plots_menu is None:
            # If no suitable menu found, add to menubar
            from PyQt5.QtWidgets import QMenu
            plots_menu = QMenu("Grid", self)
            self.menubar.addMenu(plots_menu)

        plots_menu.addSeparator()

        # Grid settings action
        self.actionGridSettings = QAction("Grid Settings...", self)
        self.actionGridSettings.triggered.connect(self.ShowGridSettings)
        plots_menu.addAction(self.actionGridSettings)

        # Export grid overlay action
        self.actionExportGridOverlay = QAction("Export Image with Grid Overlay...", self)
        self.actionExportGridOverlay.setEnabled(False)
        self.actionExportGridOverlay.triggered.connect(self.ExportGridOverlay)
        plots_menu.addAction(self.actionExportGridOverlay)

    def _setup_star_reference_menu(self):
        """Setup star reference file save/load menu actions."""
        from PyQt5.QtWidgets import QAction

        # Add to File menu
        self.menuFails.addSeparator()

        # Save stars as Alt-Az action
        self.actionSaveStarsAltAz = QAction("Save Stars as Alt-Az...", self)
        self.actionSaveStarsAltAz.setEnabled(False)
        self.actionSaveStarsAltAz.triggered.connect(self.SaveStarsAltAz)
        self.menuFails.addAction(self.actionSaveStarsAltAz)

        # Load/append stars from Alt-Az action
        self.actionLoadStarsAltAz = QAction("Load Stars from Alt-Az...", self)
        self.actionLoadStarsAltAz.setEnabled(False)
        self.actionLoadStarsAltAz.triggered.connect(self.LoadStarsAltAz)
        self.menuFails.addAction(self.actionLoadStarsAltAz)

    def load_settings_to_ui(self):
        """Load settings from app_settings into UI components"""
        if self.app_settings.last_directory:
            file_dialog_manager.last_directory = self.app_settings.last_directory

    def save_ui_to_settings(self):
        """Save UI state to app_settings and persist to file"""
        self.app_settings.last_directory = file_dialog_manager.last_directory
        self.app_settings.save_to_file()

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

        # Catalog menu items - enabled when camera is available
        self.actionShowCatalogStars.setEnabled(has_camera)
        self.actionCatalogSettings.setEnabled(has_camera)

        # Residual menu items - enabled when camera and star references are available
        has_stars = self.cloudimage is not None and len(self.cloudimage.starReferences) > 0
        has_calibration = has_camera and has_stars
        self.actionShowResidualScatter.setEnabled(has_calibration)
        self.actionShowResidualQuiver.setEnabled(has_calibration)
        self.actionTransferCatalogStars.setEnabled(has_camera and self.show_catalog_stars)

        # Star reference file menu items
        self.actionSaveStarsAltAz.setEnabled(has_camera and has_stars)
        self.actionLoadStarsAltAz.setEnabled(has_camera)

        # Grid export menu item
        self.actionExportGridOverlay.setEnabled(has_camera)

        # Auto star matching - requires camera
        self.actionAutoStarMatch.setEnabled(has_camera)

    def onUpdateText(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        self.save_ui_to_settings()
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
            # Print radial distortion
            k1 = cldim.camera.camera_enu.k1
            k2 = cldim.camera.camera_enu.k2
            k3 = cldim.camera.camera_enu.k3
            distortion_str = f'Distortion: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}'

            # Add tangential distortion if present
            p1 = getattr(cldim.camera.camera_enu, 'p1', None)
            p2 = getattr(cldim.camera.camera_enu, 'p2', None)
            if p1 is not None and p2 is not None:
                distortion_str += f', p1={p1:.6f}, p2={p2:.6f}'

            # Add rational distortion if present
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
            cldim = self.cloudimage
            cldim.PrepareCamera(**self.app_settings.camera_calibration.to_dict())
            self.PrintCameraParameters()
            self.DrawAltAzClick()
        self.update_ui_state()

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
        if camfile != '':
            self.cloudimage.LoadCamera(camfile)

    @handle_exceptions(method_name="Saving camera")
    def SaveCamera(self):
        camfile = os.path.splitext(self.cloudimage.filename)[0]+'_enu.json'
        camfile = gui_save_fname(
            directory=camfile,
            caption='Kameras fails',
            filter='(*.json)')
        if camfile != '':
            self.cloudimage.SaveCamera(camfile)

    @handle_exceptions(method_name="Drawing Alt-Az grid")
    def DrawAltAzClick(self):
        if self.cloudimage is not None:
            self.MplWidget1.canvas.initplot()
            ax = self.MplWidget1.canvas.ax
            grid_kwargs = self.grid_settings.to_grid_kwargs()
            plots_altazgrid.PlotAltAzGrid_v2(self.cloudimage.imagearray, self.cloudimage.camera.camera_enu,
                                              ax=ax, grid_kwargs=grid_kwargs)

            # Ensure axis limits are set for proper zoom/pan
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
                        p = ax.plot(ix, iy, marker='o', fillstyle='none', alpha=0)  # Invisible but get color
                    if self.show_star_names:
                        ax.annotate(sr.name, xy=(ix, iy), xytext=(
                            3, 3), color='#AAFFAA', fontsize=16, textcoords='offset pixels')
                    ax.plot(cpx[0], cpx[1], marker='x',
                            fillstyle='none', color=p[0].get_color())

            # Redraw catalog stars if visible
            if self.show_catalog_stars:
                self._ensure_catalog_overlay()
                if self.catalog_overlay is not None:
                    self.catalog_overlay.update_catalog()
                    self.catalog_overlay.draw()

            # Redraw residual overlays if visible
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

            # Redraw catalog stars if visible
            if self.show_catalog_stars:
                self._ensure_catalog_overlay()
                if self.catalog_overlay is not None:
                    self.catalog_overlay.update_catalog()
                    self.catalog_overlay.draw()

            # Redraw residual overlays if visible
            if self.show_residual_scatter or self.show_residual_quiver:
                if self._update_residual_overlay():
                    if self.show_residual_scatter:
                        self.residual_overlay.show_scatter()
                    if self.show_residual_quiver:
                        self.residual_overlay.show_quiver()

            self.MplWidget1.canvas.draw()
            return ax

    @handle_exceptions(method_name="Starting star digitization")
    def StartDigitizeStars(self):
        if self.isDigitizeStars is None:
            self.DrawImage(plot_stars=False)
            ax = self.MplWidget1.canvas.ax
            if hasattr(ax, '__len__'):
                ax = ax[0]

            # Always ensure catalog overlay is created for auto-suggest
            # (even if not visible, we need it for the suggestion feature)
            self._ensure_catalog_overlay()
            if self.catalog_overlay is not None and self.catalog_overlay.catalog_df is None:
                # Populate catalog data if not already done
                self.catalog_overlay.update_catalog()

            # Ensure residual overlay is created if we have camera and stars
            if self.cloudimage.camera is not None and len(self.cloudimage.starReferences) > 0:
                self._ensure_residual_overlay()

            self.star_digitizer = StarDigitizer(
                ax, self.cloudimage, self,
                catalog_overlay=self.catalog_overlay,
                residual_overlay=self.residual_overlay
            )
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

    @handle_exceptions()
    def CameraCalibrationParameters(self):
        """Open camera calibration parameters dialog"""
        accepted, new_params = show_camera_parameters_dialog(
            initial_params=self.app_settings.camera_calibration
        )

        if accepted:
            self.app_settings.camera_calibration = new_params
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
            self.update_ui_state()

            if hasattr(self, 'MplWidget1') and hasattr(self.MplWidget1, 'canvas'):
                if hasattr(self.MplWidget1.canvas, 'ax'):
                    self.DrawAltAzClick()
        else:
            action = 'izveide' if existing_camera is None else 'modificēšana'
            print(f'Kameras {action} atcelta')

    @handle_exceptions(method_name="Adding star with Alt-Az coordinates")
    def addStarWithAltAz(self, name, x_coord, y_coord, az_deg, alt_deg):
        """
        Add a star reference with Alt-Az coordinates.

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

    def _ensure_catalog_overlay(self):
        """Ensure catalog overlay is created and initialized."""
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            return False

        ax = self.MplWidget1.canvas.ax
        if hasattr(ax, '__len__'):
            ax = ax[0]

        # Create or update catalog overlay
        if self.catalog_overlay is None:
            self.catalog_overlay = CatalogStarOverlay(
                ax,
                self.cloudimage.camera,
                self.cloudimage.location,
                self.cloudimage.date,
                min_magnitude=self.catalog_min_magnitude,
                max_magnitude=self.catalog_max_magnitude,
                overshoot_px=self.catalog_overshoot_px,
                min_altitude=self.catalog_min_altitude
            )
        else:
            # Update existing overlay with current parameters
            self.catalog_overlay.ax = ax
            self.catalog_overlay.camera = self.cloudimage.camera
            self.catalog_overlay.location = self.cloudimage.location
            self.catalog_overlay.observation_time = self.cloudimage.date
            self.catalog_overlay.set_magnitude_range(self.catalog_min_magnitude, self.catalog_max_magnitude)
            self.catalog_overlay.set_altitude_filter(self.catalog_min_altitude)

        return True

    @handle_exceptions(method_name="Toggling catalog stars")
    def ToggleCatalogStars(self):
        """Toggle visibility of catalog stars."""
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
        """Show catalog star settings dialog."""
        if self.cloudimage is None:
            return

        accepted, (min_mag, max_mag, min_alt, overshoot) = show_catalog_settings_dialog(
            parent=self,
            min_magnitude=self.catalog_min_magnitude,
            max_magnitude=self.catalog_max_magnitude,
            min_altitude=self.catalog_min_altitude,
            overshoot_px=self.catalog_overshoot_px
        )

        if accepted:
            self.catalog_min_magnitude = min_mag
            self.catalog_max_magnitude = max_mag
            self.catalog_min_altitude = min_alt
            self.catalog_overshoot_px = overshoot

            print(f"Catalog settings: mag {min_mag} to {max_mag}, alt >= {min_alt}°, overshoot {overshoot}px")

            # Update catalog overlay if visible
            if self.show_catalog_stars and self.catalog_overlay is not None:
                self.catalog_overlay.set_magnitude_range(min_mag, max_mag)
                self.catalog_overlay.set_altitude_filter(min_alt)
                self.catalog_overlay.overshoot_px = overshoot
                self.catalog_overlay.refresh()

    @handle_exceptions(method_name="Transferring catalog stars")
    def TransferCatalogStars(self):
        """Transfer all visible catalog stars to digitization."""
        if not self._ensure_catalog_overlay():
            return

        if self.catalog_overlay.catalog_df is None or len(self.catalog_overlay.catalog_df) == 0:
            print("No catalog stars available")
            return

        # Get all visible catalog stars
        stars_to_transfer = self.catalog_overlay.get_all_visible_stars()

        if stars_to_transfer is None or len(stars_to_transfer) == 0:
            print("No catalog stars to transfer")
            return

        # Ask for confirmation
        from guihelpers.qthelper import gui_confirm
        confirm = gui_confirm(
            caption=f"Transfer {len(stars_to_transfer)} catalog stars to digitization?"
        )

        if not confirm:
            return

        # Transfer stars
        transferred_count = 0
        for _, star in stars_to_transfer.iterrows():
            # Check if star already exists
            star_exists = any(
                sr.name == star['name'] for sr in self.cloudimage.starReferences
            )

            if not star_exists:
                # Create StarReference with RA/DEC coordinates but keep catalog name
                # The RA/DEC format will be parsed and stored in skycoord
                star_ref = StarReference(
                    f"ra:{star['ra']:.6f},{star['dec']:.6f}",
                    [star['pixel_x'], star['pixel_y']]
                )
                # Override the display name to use catalog name instead of "RA/DEC(...)"
                star_ref.name = star['name']
                self.cloudimage.starReferences.append(star_ref)
                transferred_count += 1

        print(f"Transferred {transferred_count} catalog stars to digitization")
        print(f"Total digitized stars: {len(self.cloudimage.starReferences)}")

        # Refresh display
        if self.isDigitizeStars and self.star_digitizer is not None:
            self.star_digitizer._load_existing_stars()
        else:
            self.DrawImage()

    def _ensure_residual_overlay(self):
        """Ensure residual overlay is created and initialized."""
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("Cannot create residual overlay: no camera available")
            return False

        if len(self.cloudimage.starReferences) == 0:
            print("Cannot create residual overlay: no star references")
            return False

        ax = self.MplWidget1.canvas.ax
        if hasattr(ax, '__len__'):
            ax = ax[0]

        # Create residual overlay if needed
        if self.residual_overlay is None:
            from guihelpers.residual_overlay import ResidualOverlay
            self.residual_overlay = ResidualOverlay(ax)
        else:
            # Update existing overlay axis
            self.residual_overlay.ax = ax

        return True

    def _update_residual_overlay(self):
        """Calculate and update residual overlay data."""
        if not self._ensure_residual_overlay():
            return False

        # Calculate residuals from current camera calibration
        residual_data = self.cloudimage.camera.calculate_residuals(
            self.cloudimage.starReferences,
            self.cloudimage.location,
            self.cloudimage.date
        )

        if residual_data is None:
            print("Failed to calculate residuals")
            return False

        # Update overlay with new data
        self.residual_overlay.set_residuals(
            residual_data['star_pixel_coords'],
            residual_data['model_pixel_coords']
        )

        print(f"Residuals updated: RMS = {residual_data['rms']:.2f} pixels")
        return True

    @handle_exceptions(method_name="Toggling residual scatter")
    def ToggleResidualScatter(self):
        """Toggle visibility of residual scatter plot."""
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
        """Toggle visibility of residual quiver plot."""
        self.show_residual_quiver = self.actionShowResidualQuiver.isChecked()

        if not self._update_residual_overlay():
            self.actionShowResidualQuiver.setChecked(False)
            return

        if self.show_residual_quiver:
            self.residual_overlay.show_quiver()
        else:
            self.residual_overlay.hide_quiver()

        self.MplWidget1.canvas.draw()

    @handle_exceptions(method_name="Toggling star circles")
    def ToggleStarCircles(self):
        """Toggle visibility of star position circles."""
        self.show_star_circles = self.actionShowStarCircles.isChecked()
        self.DrawImage()

    @handle_exceptions(method_name="Toggling star names")
    def ToggleStarNames(self):
        """Toggle visibility of star name labels."""
        self.show_star_names = self.actionShowStarNames.isChecked()
        self.DrawImage()

    @handle_exceptions(method_name="Saving stars as Alt-Az")
    def SaveStarsAltAz(self):
        """Save star references as Alt-Az coordinates."""
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available to convert stars to Alt-Az")
            return

        if len(self.cloudimage.starReferences) == 0:
            print("No stars to save")
            return

        # Get default filename
        default_filename = os.path.splitext(self.cloudimage.filename)[0] + '_altaz.txt'

        filename = gui_save_fname(
            directory=default_filename,
            caption='Save stars as Alt-Az',
            filter='(*.txt)')

        if filename == '':
            return

        try:
            # Convert stars to Alt-Az and save
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Star references in Alt-Az format\n")
                f.write("# Name\tPixel_X\tPixel_Y\tAzimuth\tAltitude\n")

                for star_ref in self.cloudimage.starReferences:
                    # Get pixel coordinates
                    px, py = star_ref.pixelcoords

                    # Get Alt-Az coordinates
                    if star_ref.hasDirectAltAz():
                        # Already has Alt-Az
                        az = star_ref.altaz_coord.az.deg
                        alt = star_ref.altaz_coord.alt.deg
                    else:
                        # Calculate Alt-Az from sky coordinates
                        altaz = star_ref.getAltAzCoord(self.cloudimage.altaz)
                        az = altaz.az.deg
                        alt = altaz.alt.deg

                    # Write in format: name, pixel_x, pixel_y, azimuth, altitude
                    f.write(f"{star_ref.name}\t{px:.2f}\t{py:.2f}\t{az:.6f}\t{alt:.6f}\n")

            print(f"Saved {len(self.cloudimage.starReferences)} stars to {filename}")

        except Exception as e:
            print(f"Error saving stars: {e}")

    @handle_exceptions(method_name="Loading stars from Alt-Az")
    def LoadStarsAltAz(self):
        """Load and append star references from Alt-Az file."""
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available - cannot load Alt-Az stars")
            return

        filename = gui_fname(
            caption='Load stars from Alt-Az file',
            filter='(*.txt)')

        if filename == '':
            return

        try:
            loaded_count = 0
            skipped_count = 0

            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse line: name, pixel_x, pixel_y, azimuth, altitude
                    parts = line.split('\t')
                    if len(parts) < 5:
                        # Try space-separated
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


                        # Create star reference with Alt-Az coordinates
                        star_ref = StarReference(f"{az:.6f},{alt:.6f}", [px, py])
                        star_ref.name = f'AZ:{name}'  # Override name

                        self.cloudimage.starReferences.append(star_ref)
                        loaded_count += 1

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line '{line}': {e}")
                        skipped_count += 1

            print(f"Loaded {loaded_count} stars from {filename}")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} stars (duplicates or invalid)")
            print(f"Total stars: {len(self.cloudimage.starReferences)}")

            # Refresh display
            if self.isDigitizeStars and self.star_digitizer is not None:
                self.star_digitizer._load_existing_stars()
            else:
                self.DrawImage()

        except Exception as e:
            print(f"Error loading stars: {e}")

    @handle_exceptions(method_name="Showing grid settings")
    def ShowGridSettings(self):
        """Show grid settings dialog."""
        accepted, new_settings = show_grid_settings_dialog(
            parent=self,
            settings=self.grid_settings
        )

        if accepted:
            self.grid_settings = new_settings
            print("Grid settings updated")

            # Redraw if currently showing grid
            if hasattr(self, 'MplWidget1') and hasattr(self.MplWidget1, 'canvas'):
                if hasattr(self.MplWidget1.canvas, 'ax'):
                    current_children = self.MplWidget1.canvas.ax.get_children()
                    # Check if we have contour plots (grid is being displayed)
                    has_contours = any('QuadContourSet' in str(type(child)) for child in current_children)
                    if has_contours:
                        self.DrawAltAzClick()

    @handle_exceptions(method_name="Exporting grid overlay")
    def ExportGridOverlay(self):
        """Export image with grid overlay to file."""
        if self.cloudimage is None or not hasattr(self.cloudimage, 'camera') or self.cloudimage.camera is None:
            print("No camera available to export grid")
            return

        # Get default filename
        default_filename = os.path.splitext(self.cloudimage.filename)[0] + '_grid.jpg'

        filename = gui_save_fname(
            directory=default_filename,
            caption='Export Image with Grid Overlay',
            filter='(*.jpg *.png)')

        if filename == '':
            return

        try:
            # Get grid settings
            grid_kwargs = self.grid_settings.to_grid_kwargs()

            # Create and save grid overlay figure
            plots.CreateGridOverlayFigure(
                self.cloudimage,
                grid_kwargs=grid_kwargs,
                dpi=100,
                filename=filename,
                close_figure=True
            )

            h_px, w_px = self.cloudimage.imagearray.shape[:2]
            print(f"Exported grid overlay to {filename}")
            print(f"Image size: {w_px}x{h_px} pixels")

        except Exception as e:
            print(f"Error exporting grid overlay: {e}")
            import traceback
            traceback.print_exc()


    def _setup_auto_calibration_menu(self):
        """Add automatic star matching actions to the Darbības menu."""
        from PyQt5.QtWidgets import QAction

        self.menuZ_m_t.addSeparator()

        self.actionAutoStarMatch = QAction("Auto Star Matching...", self)
        self.actionAutoStarMatch.setEnabled(False)
        self.actionAutoStarMatch.triggered.connect(self.AutoStarMatch)
        self.menuZ_m_t.addAction(self.actionAutoStarMatch)

        self.actionAutoMatchSettings = QAction("Auto Star Matching Settings...", self)
        self.actionAutoMatchSettings.triggered.connect(self.ShowAutoMatchSettings)
        self.menuZ_m_t.addAction(self.actionAutoMatchSettings)

    @handle_exceptions(method_name="Auto star matching")
    def AutoStarMatch(self):
        """Detect stars in image, match to catalog, fill star references, update camera."""
        import sys
        import os

        if self.cloudimage is None or self.cloudimage.camera is None:
            print("No camera loaded - cannot run auto star matching")
            return

        star_extract_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'sudrabainiemakoni', 'calibration', 'StarExtract'
        ))
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
            # refined_camera_enu, results = calibrate_camera_pose(
            #     detected_xy,
            #     self.cloudimage.camera.camera_enu,
            #     self.cloudimage.location,
            #     self.cloudimage.date,
            #     max_magnitude_coarse     = s.max_magnitude_coarse,
            #     angle_tol_deg            = s.angle_tol_deg,
            #     focal_length_uncertainty = s.focal_length_uncertainty,
            #     n_search                 = s.n_search,
            #     max_magnitude_fine       = s.max_magnitude_fine,
            #     nn_max_dist_px           = s.nn_max_dist_px,
            #     optimize_intrinsics      = s.optimize_intrinsics,
            #     max_rms_coarse_px        = s.max_rms_coarse_px,
            #     max_rms_fine_px          = s.max_rms_fine_px,
            #     debug_star = 'Leo 60'
            # )
            print(self.cloudimage.camera.camera_enu)
            refined_camera_enu, results = calibrate_camera_pose(
                    detected_xy,
                    self.cloudimage.camera.camera_enu,
                    self.cloudimage.location,
                    self.cloudimage.date,
                    max_magnitude_fine=5.5,
                    angle_tol_deg=0.1,
                    debug_star='Leo 60'
            )


        except StarMatchError as e:
            print(f"Auto star matching failed: {e}")
            return

        rms = results['rms_px']
        n_stars = len(results['detected_xy'])
        print(f"Calibration succeeded: {n_stars} stars, RMS = {rms:.2f} px")

        if s.update_camera:
            # Build refined Camera using from_manual_parameters, preserving existing
            # projection type, distortion type and coefficients.
            from sudrabainiemakoni.cloudimage_camera import Camera
            from sudrabainiemakoni import cameraprojections
            from sudrabainiemakoni.cloudimage_camera import name_by_distortion

            old_enu = self.cloudimage.camera.camera_enu
            image_size = self.cloudimage.camera.image_size
            proj_name = cameraprojections.name_by_projection(old_enu.projection)
            dist_name = name_by_distortion(old_enu.lens)

            new_camera = Camera.from_manual_parameters(
                image_size       = image_size,
                location         = self.cloudimage.location,
                fx               = refined_camera_enu.focallength_x_px,
                fy               = refined_camera_enu.focallength_y_px,
                cx               = refined_camera_enu.center_x_px,
                cy               = refined_camera_enu.center_y_px,
                azimuth          = refined_camera_enu.heading_deg,
                elevation        = refined_camera_enu.tilt_deg - 90,
                rotation         = refined_camera_enu.roll_deg,
                k1               = old_enu.k1,
                k2               = old_enu.k2,
                k3               = old_enu.k3,
                p1               = getattr(old_enu, 'p1', 0),
                p2               = getattr(old_enu, 'p2', 0),
                k4               = getattr(old_enu, 'k4', 0),
                k5               = getattr(old_enu, 'k5', 0),
                k6               = getattr(old_enu, 'k6', 0),
                projection       = proj_name,
                distortion_type  = dist_name,
            )
            self.cloudimage.camera = new_camera
            print("Camera updated")
        else:
            print("Camera not updated (update_camera=False)")

        # Replace star references with RA/DEC from matched catalog rows
        self.cloudimage.starReferences = []
        catalog_rows = results['catalog_rows']
        matched_det_xy = results['detected_xy']

        for i in range(len(matched_det_xy)):
            px, py = matched_det_xy[i]
            row = catalog_rows.iloc[i]
            star = StarReference(
                f"ra:{float(row['ra']):.6f},{float(row['dec']):.6f}",
                [float(px), float(py)],
            )
            star.name = str(row['name'])
            self.cloudimage.starReferences.append(star)

        print(f"Added {len(matched_det_xy)} star references")
        self.update_ui_state()
        self.DrawAltAzClick()

    @handle_exceptions(method_name="Auto match settings")
    def ShowAutoMatchSettings(self):
        """Show the auto star matching settings dialog."""
        from guihelpers.auto_match_settings_dialog import show_auto_match_settings_dialog

        accepted, new_settings = show_auto_match_settings_dialog(
            parent=self,
            settings=self.app_settings.auto_match,
        )

        if accepted:
            self.app_settings.auto_match = new_settings
            self.app_settings.save_to_file()
            print(f"Auto match settings saved")


def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler to prevent GUI termination on exceptions."""
    import traceback
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("EXCEPTION CAUGHT:")
    print(tb)


def main():
    """Main entry point for the star referencing GUI application."""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("ERROR: PyQt5 is not installed.")
        print("Please install GUI dependencies with: pip install sudrabainiemakoni[gui]")
        sys.exit(1)

    sys.excepthook = excepthook

    try:
        app = QApplication(sys.argv)
        myapp = StarReferenceWindow()
        myapp.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        import os
        error_log = os.path.join(os.path.dirname(__file__), 'star_reference_error.log')
        with open(error_log, 'w') as f:
            f.write("ERROR starting GUI application:\n")
            f.write(traceback.format_exc())
        print("ERROR starting GUI application:")
        print(traceback.format_exc())
        print(f"Error logged to: {error_log}")
        sys.exit(1)


if __name__ == '__main__':
    main()
