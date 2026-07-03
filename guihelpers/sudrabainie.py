import sys
import os
import numpy as np
import tilemapbase
from sudrabainiemakoni.cloudimage import CloudImage, CloudImagePair, HeightMap
from sudrabainiemakoni import plots, utils
from sudrabainiemakoni.webmercatorimage import ProjectionImageWebMercator

try:
    from .base_window import BaseCloudImageWindow, excepthook
    from .smgui import Ui_MainWindow
    from .exceptions import handle_exceptions
    from .control_point_digitizer import ControlPointDigitizer
except ImportError:
    from base_window import BaseCloudImageWindow, excepthook
    from smgui import Ui_MainWindow
    from exceptions import handle_exceptions
    from control_point_digitizer import ControlPointDigitizer

# Import via absolute path to guarantee the same module instance as base_window.py
from guihelpers.qthelper import gui_fname, gui_save_fname, gui_string

# pyuic5 smgui.ui -o smgui.py


class MainW(BaseCloudImageWindow, Ui_MainWindow):
    def __init__(self):
        # Extra state before super().__init__ connects menus/actions
        self.cloudimage2 = None
        self.cpair = None
        self.projHeight = 80  # km
        self.map_bounds = [17, 33, 56, 63]
        self.map_alpha = 0.85
        self.heightmap = None
        self.projected_image = None
        self.isDigitizeControlPoints = None
        self.measure_events = None
        self.z1, self.z2 = 75, 90
        self.control_point_digitizer = None
        self.measure = None

        super().__init__()

        # Additional menu connections not in base
        self.actionIelas_t_otro_projektu.triggered.connect(self.LoadProject2)
        self.actionProjic_t.triggered.connect(lambda: self.ProjectClick(tips=0))
        self.actionProjic_t_kop.triggered.connect(lambda: self.ProjectClick(tips=2))
        self.actionProjic_t_no_augstumu_kartes.triggered.connect(lambda: self.ProjectFromMapClick(tips=0))
        self.actionProjic_t_kop_no_augstumu_kartes.triggered.connect(lambda: self.ProjectFromMapClick(tips=2))
        self.actionProjekcijas_apgabals.triggered.connect(self.ChangeRegion)
        self.actionKartes_apgabals.triggered.connect(self.MapRegion)
        self.actionKontrolpunkti.triggered.connect(self.DigitizeControlPointsClick)
        self.actionIelas_t_kontrolpunktus.triggered.connect(self.LoadControlPoints)
        self.actionSaglab_t_augstumu_punktus_CSV.triggered.connect(self.SaveHeightPointsCSV)
        self.actionKontrolpunktu_augstumus.triggered.connect(self.DrawControlPointHeights)
        self.actionIzveidot_augstumu_karti.triggered.connect(self.CreateHeightMap)
        self.actionIelas_t_augstumu_karti.triggered.connect(self.LoadHeightMap)
        self.actionSaglab_t_augstumu_karti.triggered.connect(self.SaveHeightMap)
        self.actionAugstumu_karti.triggered.connect(self.DrawHeightMap)
        self.actionSaglab_t_projic_to_att_lu_JPG.triggered.connect(lambda: self.SaveProjectedImage(jpg=True))
        self.actionSaglab_t_projic_to_att_lu_TIFF.triggered.connect(lambda: self.SaveProjectedImage(jpg=False))
        self.actionEpil_niju_augstums.triggered.connect(self.SetEpilineHeight)

        self.webmerc = ProjectionImageWebMercator(self.cloudimage, 17, 33, 56, 63, 1.0)

    # -- Overrides -----------------------------------------------------------

    @handle_exceptions(method_name="Loading image")
    def LoadImage(self, filename_jpg):
        self.cloudimage2 = None
        super().LoadImage(filename_jpg)

    @handle_exceptions(method_name="Loading project")
    def LoadProject(self):
        projfile = gui_fname(caption='Projekta fails', filter='(*.proj)')
        if projfile != '':
            print(f'Loading project {projfile}')
            self.cloudimage2 = None
            self.cpair = None
            self.console.clear()
            self.cloudimage = CloudImage.load(projfile)
            self.isDigitizeStars = None
            print(f'Loaded project file {projfile}')
            print(self.cloudimage)
            self.DrawImage()
        self.update_ui_state()

    def update_ui_state(self):
        super().update_ui_state()
        has_camera = self.cloudimage is not None and hasattr(self.cloudimage, 'camera') and self.cloudimage.camera is not None
        self.actionIelas_t_otro_projektu.setEnabled(self.cloudimage is not None)
        self.actionProjic_t.setEnabled(has_camera)
        self.actionProjic_t_kop.setEnabled(has_camera and self.cloudimage2 is not None)
        self.actionKontrolpunkti.setEnabled(self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionIelas_t_kontrolpunktus.setEnabled(self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionSaglab_t_augstumu_punktus_CSV.setEnabled(self.cpair is not None)
        self.actionKontrolpunktu_augstumus.setEnabled(self.cpair is not None)
        self.actionIzveidot_augstumu_karti.setEnabled(self.cpair is not None)
        self.actionIelas_t_augstumu_karti.setEnabled(self.cloudimage is not None)
        self.actionSaglab_t_augstumu_karti.setEnabled(self.heightmap is not None)
        self.actionAugstumu_karti.setEnabled(self.heightmap is not None)
        self.actionProjic_t_no_augstumu_kartes.setEnabled(self.heightmap is not None and self.cloudimage is not None)
        self.actionProjic_t_kop_no_augstumu_kartes.setEnabled(
            self.heightmap is not None and self.cloudimage is not None and self.cloudimage2 is not None)
        self.actionSaglab_t_projic_to_att_lu_JPG.setEnabled(self.projected_image is not None)
        self.actionSaglab_t_projic_to_att_lu_TIFF.setEnabled(self.projected_image is not None)

    @handle_exceptions(method_name="Drawing image")
    def DrawImageClick(self):
        self.DrawImage(otrs=self.cloudimage2 is not None)

    @handle_exceptions(method_name="Drawing image")
    def DrawImage(self, otrs=False, kontrolpunkti=False, plot_stars=True):
        self.disconnect_measurement()
        if self.cloudimage is not None:
            if otrs and self.cloudimage2 is not None:
                self.MplWidget1.canvas.initplot([121, 122])
                ax = self.MplWidget1.canvas.ax[0]
                ax2 = self.MplWidget1.canvas.ax[1]
            else:
                self.MplWidget1.canvas.initplot()
                ax = self.MplWidget1.canvas.ax
            ax.imshow(self.cloudimage.imagearray)
            if kontrolpunkti:
                self.plot_matches(ax, 0)
            elif plot_stars:
                plots.PlotStars(self.cloudimage, ax,
                                show_circles=self.show_star_circles,
                                show_names=self.show_star_names)
            if otrs and self.cloudimage2 is not None:
                ax2.imshow(self.cloudimage2.imagearray)
                if kontrolpunkti:
                    self.plot_matches(ax2, 1)
                elif plot_stars:
                    plots.PlotStars(self.cloudimage2, ax2)
            self.MplWidget1.canvas.draw()
            return ax

    @handle_exceptions(method_name="Drawing Alt-Az grid")
    def DrawAltAzClick(self):
        self.disconnect_measurement()
        super().DrawAltAzClick()

    # -- Second image --------------------------------------------------------

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

    # -- Epiline height ------------------------------------------------------

    @handle_exceptions(method_name="Setting epiline height")
    def SetEpilineHeight(self):
        s = f'{self.z1:.0f},{self.z2:.0f}'
        s = gui_string(text=s, caption='Ievadi z1,z2')
        if s is not None:
            s = s.split(',')
            self.z1, self.z2 = float(s[0]), float(s[1])
            if self.control_point_digitizer is not None:
                self.control_point_digitizer.set_height_range(self.z1, self.z2)

    # -- Control point digitization ------------------------------------------

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
            ax1 = self.MplWidget1.canvas.ax[0]
            ax2 = self.MplWidget1.canvas.ax[1]
            if self.cpair is None:
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
            self.control_point_digitizer = ControlPointDigitizer(
                ax1, ax2, self.cpair, self, self.z1, self.z2)
            self.control_point_digitizer.start_digitization()
            self.isDigitizeControlPoints = True

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
            matchfile = gui_fname(directory=matchfile, caption='Atbilstību fails', filter='(*.txt)')
            if matchfile != '':
                self.cpair = CloudImagePair(self.cloudimage, self.cloudimage2)
                self.cpair.LoadCorrespondances(matchfile)
                self.DrawImage(otrs=True, kontrolpunkti=True)
        self.update_ui_state()

    @handle_exceptions(method_name="Saving height points to CSV")
    def SaveHeightPointsCSV(self):
        if self.cpair is not None:
            llh, rayminimaldistance, z_intrinsic_error, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            csvfile = os.path.splitext(self.cloudimage.filename)[0] + '_heights.csv'
            csvfile = gui_save_fname(directory=csvfile, caption='Augstumu punktu CSV fails', filter='(*.csv)')
            if csvfile != '':
                import pandas as pd
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
                df.to_csv(csvfile, index=False, float_format='%.6f')
                print(f'Height points saved to: {csvfile}')
                print(f'Total points: {len(llh[0])}, Valid points: {sum(valid)}')

    def plot_matches(self, ax, pairNo):
        if self.cpair is not None:
            for i, (x, y) in enumerate(self.cpair.correspondances[pairNo]):
                ax.plot(x, y, marker='o', fillstyle='none', markeredgecolor='red')
                ax.annotate(str(i + 1), xy=(x, y), xytext=(3, 3),
                            color='#AAFFAA', fontsize=16, textcoords='offset pixels')

    @handle_exceptions(method_name="Drawing control point heights")
    def DrawControlPointHeights(self):
        self.disconnect_measurement()
        if self.cpair is not None:
            self.MplWidget1.canvas.initplot([121, 122])
            ax = self.MplWidget1.canvas.ax[0]
            ax2 = self.MplWidget1.canvas.ax[1]
            z1, z2 = self.z1, self.z2
            llh, _, _, _ = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            epilines = self.cpair.GetEpilinesAtHeightInterval([z1, z2], self.cpair.correspondances[1], False)
            plots.PlotValidHeightPoints(self.cloudimage.imagearray, epilines,
                                        self.cpair.correspondances[0], llh[2], None, ax=ax)
            epilines = self.cpair.GetEpilinesAtHeightInterval([z1, z2], self.cpair.correspondances[0], True)
            plots.PlotValidHeightPoints(self.cloudimage2.imagearray, epilines,
                                        self.cpair.correspondances[1], llh[2], None, ax=ax2)
            self.MplWidget1.canvas.draw()

    # -- Height map ----------------------------------------------------------

    @handle_exceptions(method_name="Creating height map")
    def CreateHeightMap(self):
        if self.cpair is not None:
            llh, _, _, valid = self.cpair.GetHeightPoints(
                *self.cpair.correspondances)
            valid[:] = True
            if any(valid):
                self.webmerc.cloudimage = self.cloudimage
                heightgrid = self.webmerc.PrepareHeightMap(llh[1][valid], llh[0][valid], llh[2][valid])
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
            projfile = os.path.splitext(self.cloudimage.filename)[0] + '.hmp'
            projfile = gui_save_fname(directory=projfile, caption='Augstumu kartes fails', filter='(*.hmp)')
            if projfile != '':
                self.heightmap.save(projfile)
                print(f'Saved heightmap {projfile}')

    @handle_exceptions(method_name="Loading height map")
    def LoadHeightMap(self):
        if self.cloudimage is not None:
            projfile = os.path.splitext(self.cloudimage.filename)[0] + '.hmp'
            projfile = gui_fname(directory=projfile, caption='Augstumu kartes fails', filter='(*.hmp)')
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
            csl = plots.PlotReferencedImages(
                self.webmerc, [self.heightmap.heightmap], camera_points=[],
                outputFileName=None,
                lonmin=self.map_bounds[0], lonmax=self.map_bounds[1],
                latmin=self.map_bounds[2], latmax=self.map_bounds[3],
                showplot=True, alpha=0.8, ax=ax)
            llh = self.heightmap.points
            valid = self.heightmap.validpoints
            xy = np.array([tilemapbase.project(lon, lat)
                           for lon, lat in zip(llh[1][valid], llh[0][valid])])
            ax.scatter(xy[:, 0], xy[:, 1], c=llh[2][valid], norm=csl[0].norm, cmap=csl[0].cmap)
            ax.figure.colorbar(csl[0])
            self.connect_measurement()
            self.MplWidget1.canvas.draw()

    # -- Projection ----------------------------------------------------------

    @handle_exceptions(method_name="Projection click")
    def ProjectClick(self, tips=0):
        if hasattr(self.cloudimage, "camera"):
            s = gui_string(text=f'{self.projHeight}', caption='Augstums kilometros')
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
                self.Project(self.heightmap.heightmap / 1000.0, atseviski=tips == 0)
            else:
                self.ProjectAveragedImage(self.heightmap.heightmap / 1000.0)

    @handle_exceptions(method_name="Projecting image")
    def Project(self, projHeight, atseviski=True):
        self.disconnect_measurement()
        self.webmerc.cloudImage = self.cloudimage
        self.webmerc.prepare_reproject_from_camera(projHeight)
        projected_image = self.webmerc.Fill_projectedImageMasked()
        self.projected_image = (self.webmerc.__getstate__(), projected_image, projHeight)
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
            self.webmerc.cloudImage = self.cloudimage2
            self.webmerc.prepare_reproject_from_camera(projHeight)
            projected_image2 = self.webmerc.Fill_projectedImage()
            self.webmerc.cloudImage = self.cloudimage
            img_mean, img_diff, _, img_bicolor = utils.getAverageImages([projected_image, projected_image2])
            self.MplWidget1.canvas.initplot([131, 132, 133])
            self.plot_projection([img_mean], self.MplWidget1.canvas.ax[0], plotMap=False, plotPoints=False)
            self.plot_projection([img_diff], self.MplWidget1.canvas.ax[1], plotMap=False, plotPoints=False)
            self.plot_projection([img_bicolor[(0, 1)]], self.MplWidget1.canvas.ax[2], plotMap=False, plotPoints=False)
            self.MplWidget1.canvas.draw()
            self.connect_measurement()

    @handle_exceptions(method_name="Changing projection region")
    def ChangeRegion(self):
        w = self.webmerc
        text = f'{w.lonmin},{w.lonmax},{w.latmin},{w.latmax},{w.pixel_per_km}'
        s = gui_string(text=text, caption='lonmin,lonmax,latmin,latmax,resolution_km')
        if s is not None:
            s = [float(x) for x in s.split(',')]
            if len(s) == 5:
                self.webmerc = ProjectionImageWebMercator(self.cloudimage, *s)
                print(self.webmerc)

    @handle_exceptions(method_name="Setting map region")
    def MapRegion(self):
        w = self.map_bounds
        text = f'{w[0]},{w[1]},{w[2]},{w[3]},{self.map_alpha}'
        s = gui_string(text=text, caption='lonmin,lonmax,latmin,latmax,map_alpha')
        if s is not None:
            s = [float(x) for x in s.split(',')]
            if len(s) == 5:
                self.map_bounds = s[:4]
                self.map_alpha = max(min(s[4], 1.0), 0.0)

    @handle_exceptions(method_name="Plotting projection")
    def plot_projection(self, pimages, ax, plotMap=True, plotPoints=True):
        def xy_latlon_str(x, y):
            lon, lat = tilemapbase.to_lonlat(x, y)
            return f'{lat:.3f}, {lon:.3f}'
        ax.format_coord = xy_latlon_str
        self.measure = None
        pp = [[self.cloudimage.location.lon.value, self.cloudimage.location.lat.value]]
        if self.cloudimage2 is not None:
            pp.append([self.cloudimage2.location.lon.value, self.cloudimage2.location.lat.value])
        plots.PlotReferencedImages(
            self.webmerc, pimages,
            camera_points=pp if plotPoints else [],
            outputFileName=None,
            lonmin=self.map_bounds[0], lonmax=self.map_bounds[1],
            latmin=self.map_bounds[2], latmax=self.map_bounds[3],
            alpha=self.map_alpha, ax=ax, plotMap=plotMap)

    @handle_exceptions(method_name="Saving projected image")
    def SaveProjectedImage(self, jpg=True):
        if self.projected_image is not None:
            ext = '.jpg' if jpg else '.tif'
            extjgw = '.jgw' if jpg else '.tfw'
            f = os.path.split(self.cloudimage.filename)
            try:
                z = float(self.projected_image[2])
                zs = f'_{z:.1f}'
            except Exception:
                zs = ''
            projfile = f[0] + '/proj_' + os.path.splitext(f[1])[0] + zs + ext
            projfile = gui_save_fname(directory=projfile, caption='Projicētais attēls', filter=f'(*{ext})')
            if projfile != '':
                jgwfile = os.path.splitext(projfile)[0] + extjgw
                wm = ProjectionImageWebMercator(None, 17, 33, 56, 63, 1.0)
                wm.__setstate__(self.projected_image[0])
                if jpg:
                    img = self.projected_image[1][:, :, 0:3]
                    wm.SaveJgw(jgwfile)
                    import imageio.v3 as iio
                    iio.imwrite(projfile, img)
                    print('Fails saglabāts:', projfile)
                else:
                    img = self.projected_image[1]
                    try:
                        wm.SaveGeoTiffRasterio(img, projfile)
                        print('GeoTIFF fails ar CRS informāciju saglabāts:', projfile)
                    except (ImportError, Exception) as e:
                        wm.SaveJgw(jgwfile)
                        import imageio.v3 as iio
                        iio.imwrite(projfile, img)
                        if isinstance(e, ImportError):
                            print('TIFF fails ar world file saglabāts (rasterio nav pieejams):', projfile)
                        else:
                            print(f'TIFF fails ar world file saglabāts (rasterio kļūda: {e}):', projfile)
                    from sudrabainiemakoni import savekml
                    savekml.mapOverlay(wm, img, self.projHeight, projfile,
                                       saveimage=False, cloudimage=self.cloudimage)

    # -- Distance measurement ------------------------------------------------

    @handle_exceptions(method_name="Disconnecting measurement")
    def disconnect_measurement(self):
        if self.measure_events is not None:
            for e in self.measure_events:
                self.MplWidget1.canvas.mpl_disconnect(e)
            self.measure_events = None

    @handle_exceptions(method_name="Connecting measurement")
    def connect_measurement(self):
        self.measure_events = [
            self.MplWidget1.canvas.mpl_connect('button_press_event', self.onclick_measure_distance),
            self.MplWidget1.canvas.mpl_connect('motion_notify_event', self.move_measure_distance)]

    @handle_exceptions(method_name="Calculating distance")
    def distance(self, p1_webmerc, p2_webmerc):
        ll1 = tilemapbase.to_lonlat(*p1_webmerc)
        ll2 = tilemapbase.to_lonlat(*p2_webmerc)
        import pymap3d.vincenty
        ll1 = ll1[1], ll1[0]
        ll2 = ll2[1], ll2[0]
        dist, az = pymap3d.vincenty.vdist(*ll1, *ll2)
        return dist

    @handle_exceptions(method_name="Distance measurement click")
    def onclick_measure_distance(self, event):
        ax = event.inaxes
        if ax is None:
            return
        try:
            zooming_panning = (ax.figure.canvas.cursor().shape() not in [0, 13])
        except Exception:
            zooming_panning = False
        if zooming_panning:
            return
        if event.button == 1:
            if self.measure is None:
                pp = ax.plot(event.xdata, event.ydata, marker='D', ms=5, color='orange')
                self.measure = {'p1': (event.xdata, event.ydata), 'plotp1': pp}
                ax.figure.canvas.draw()
            else:
                xx = (event.xdata, event.ydata)
                dist = self.distance(self.measure['p1'], xx)
                for c in ['plotp1', 'ln']:
                    if c in self.measure:
                        self.measure[c][0].remove()
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
                    self.measure['ln'].pop(0).remove()
                if 'annot' in self.measure:
                    self.measure['annot'].remove()
                self.measure['ln'] = ax.plot(
                    [self.measure['p1'][0], xx[0]], [self.measure['p1'][1], xx[1]], color='black')
                self.measure['annot'] = ax.annotate(
                    f'{dist:.0f}m', xy=xx, xytext=(10, 10), textcoords='offset points',
                    fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
                ax.figure.canvas.draw()


def main():
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("ERROR: PyQt5 is not installed.")
        print("Please install GUI dependencies with: pip install sudrabainiemakoni[gui]")
        sys.exit(1)

    sys.excepthook = excepthook

    try:
        app = QApplication(sys.argv)
        myapp = MainW()
        myapp.show()
        sys.exit(app.exec_())
    except Exception:
        import traceback
        error_log = os.path.join(os.path.dirname(__file__), 'sudrabainie_error.log')
        with open(error_log, 'w') as f:
            f.write("ERROR starting GUI application:\n")
            f.write(traceback.format_exc())
        print("ERROR starting GUI application:")
        print(traceback.format_exc())
        print(f"Error logged to: {error_log}")
        sys.exit(1)


if __name__ == '__main__':
    main()
