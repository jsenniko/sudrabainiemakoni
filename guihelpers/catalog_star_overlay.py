"""
CatalogStarOverlay - Manages visualization of catalog stars on astronomical images.
This class provides independent catalog star display with magnitude filtering and
selection box functionality for bulk transfer to digitization.
"""

import numpy as np
import pandas as pd
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.text import Text


class CatalogStarOverlay:
    """
    Manages the display of catalog stars as a matplotlib overlay.

    This class is independent of StarDigitizer and can be used both during
    digitization and for general visualization (e.g., with Alt-Az grid).
    """

    def __init__(self, ax, camera, location, observation_time,
                 min_magnitude=-3, max_magnitude=4, overshoot_px=20, min_altitude=0.0):
        """
        Initialize the catalog star overlay.

        Args:
            ax: matplotlib axes object
            camera: Camera object with calibration
            location: astropy.coordinates.EarthLocation
            observation_time: astropy.time.Time
            min_magnitude: Minimum magnitude to display (default: -3)
            max_magnitude: Maximum magnitude to display (default: 4)
            overshoot_px: Pixels beyond image edge to include (default: 20)
            min_altitude: Minimum altitude in degrees (default: 0.0)
        """
        self.ax = ax
        self.camera = camera
        self.location = location
        self.observation_time = observation_time
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.overshoot_px = overshoot_px
        self.min_altitude = min_altitude

        # Catalog star data and artists
        self.catalog_df = None
        self.artists = []  # matplotlib artists for catalog stars
        self.visible = False

        # Selection box for bulk operations
        self.selection_box = None
        self.selection_box_artist = None

    def set_magnitude_range(self, min_magnitude, max_magnitude):
        """Update magnitude filter range."""
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude

    def set_altitude_filter(self, min_altitude):
        """Update minimum altitude filter."""
        self.min_altitude = min_altitude

    def update_catalog(self):
        """
        Update catalog star data by projecting stars onto the image.
        This should be called when camera, location, or time changes.
        """
        from sudrabainiemakoni.calibration.catalog_star_projection import project_catalog_stars

        if self.camera is None:
            print("Cannot update catalog: no camera available")
            self.catalog_df = None
            return

        try:
            self.catalog_df = project_catalog_stars(
                self.camera,
                self.location,
                self.observation_time,
                overshoot_px=self.overshoot_px,
                min_magnitude=self.min_magnitude,
                max_magnitude=self.max_magnitude,
                min_altitude=self.min_altitude
            )
            print(f"Catalog updated: {len(self.catalog_df)} stars found")
        except Exception as e:
            print(f"Error updating catalog: {e}")
            self.catalog_df = None

    def show(self):
        """Show catalog stars on the axes."""
        if not self.visible:
            self.visible = True
            self.draw()

    def hide(self):
        """Hide catalog stars from the axes."""
        if self.visible:
            self.visible = False
            self._clear_artists()
            self.ax.figure.canvas.draw_idle()

    def toggle(self):
        """Toggle visibility of catalog stars."""
        if self.visible:
            self.hide()
        else:
            self.show()

    def draw(self):
        """Draw catalog stars on the axes."""
        self._clear_artists()

        if not self.visible:
            return

        if self.catalog_df is None or len(self.catalog_df) == 0:
            print("No catalog stars to draw")
            return

        # Filter by current magnitude range (in case it changed after update)
        filtered_df = self.catalog_df[
            (self.catalog_df['mag'] >= self.min_magnitude) &
            (self.catalog_df['mag'] <= self.max_magnitude)
        ]

        if len(filtered_df) == 0:
            print(f"No stars in magnitude range {self.min_magnitude} to {self.max_magnitude}")
            return

        # Draw each catalog star
        for _, star in filtered_df.iterrows():
            x, y = star['pixel_x'], star['pixel_y']
            name = star['name']
            mag = star['mag']

            # Draw marker (use + symbol to distinguish from digitized stars)
            marker = self.ax.plot(
                x, y,
                marker='+',
                markersize=10,
                markeredgewidth=2,
                color='cyan',
                alpha=0.7
            )[0]
            self.artists.append(marker)

            # Draw label with magnitude
            label = self.ax.annotate(
                f"{name} ({mag:.1f})",
                xy=(x, y),
                xytext=(5, 5),
                textcoords='offset pixels',
                color='cyan',
                fontsize=8,
                alpha=0.7
            )
            self.artists.append(label)

        print(f"Drew {len(filtered_df)} catalog stars")
        self.ax.figure.canvas.draw_idle()

    def _clear_artists(self):
        """Remove all catalog star artists from the axes."""
        for artist in self.artists:
            try:
                artist.remove()
            except:
                pass
        self.artists.clear()

        # Also clear selection box if present
        if self.selection_box_artist is not None:
            try:
                self.selection_box_artist.remove()
            except:
                pass
            self.selection_box_artist = None

    def set_selection_box(self, x_min, y_min, x_max, y_max):
        """
        Set a selection box for bulk operations.

        Args:
            x_min, y_min: Top-left corner in pixel coordinates
            x_max, y_max: Bottom-right corner in pixel coordinates
        """
        self.selection_box = (x_min, y_min, x_max, y_max)
        self._draw_selection_box()

    def clear_selection_box(self):
        """Clear the selection box."""
        self.selection_box = None
        if self.selection_box_artist is not None:
            try:
                self.selection_box_artist.remove()
            except:
                pass
            self.selection_box_artist = None
            self.ax.figure.canvas.draw_idle()

    def _draw_selection_box(self):
        """Draw the selection box on the axes."""
        if self.selection_box_artist is not None:
            try:
                self.selection_box_artist.remove()
            except:
                pass

        if self.selection_box is None:
            return

        x_min, y_min, x_max, y_max = self.selection_box
        width = x_max - x_min
        height = y_max - y_min

        self.selection_box_artist = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor='yellow',
            facecolor='none',
            linestyle='--',
            alpha=0.8
        )
        self.ax.add_patch(self.selection_box_artist)
        self.ax.figure.canvas.draw_idle()

    def get_stars_in_selection(self):
        """
        Get catalog stars within the current selection box.

        Returns:
            pandas.DataFrame: Filtered catalog stars within selection box,
                            or None if no selection box is set
        """
        if self.selection_box is None:
            return None

        if self.catalog_df is None or len(self.catalog_df) == 0:
            return None

        x_min, y_min, x_max, y_max = self.selection_box

        # Ensure correct ordering
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        # Filter stars within box
        in_box = (
            (self.catalog_df['pixel_x'] >= x_min) &
            (self.catalog_df['pixel_x'] <= x_max) &
            (self.catalog_df['pixel_y'] >= y_min) &
            (self.catalog_df['pixel_y'] <= y_max) &
            (self.catalog_df['mag'] >= self.min_magnitude) &
            (self.catalog_df['mag'] <= self.max_magnitude)
        )

        return self.catalog_df[in_box].copy()

    def get_all_visible_stars(self):
        """
        Get all visible catalog stars (filtered by magnitude).

        Returns:
            pandas.DataFrame: All visible catalog stars
        """
        if self.catalog_df is None or len(self.catalog_df) == 0:
            return None

        filtered = self.catalog_df[
            (self.catalog_df['mag'] >= self.min_magnitude) &
            (self.catalog_df['mag'] <= self.max_magnitude)
        ]

        return filtered.copy()

    def find_nearest_star(self, x, y, max_distance=20):
        """
        Find the nearest catalog star to given pixel coordinates.
        Useful for auto-suggest functionality during digitization.

        Args:
            x, y: Pixel coordinates
            max_distance: Maximum distance in pixels to consider (default: 20)

        Returns:
            dict: Star data with keys 'name', 'mag', 'distance', etc., or None
        """
        if self.catalog_df is None or len(self.catalog_df) == 0:
            return None

        # Calculate distances to all catalog stars
        distances = np.sqrt(
            (self.catalog_df['pixel_x'] - x)**2 +
            (self.catalog_df['pixel_y'] - y)**2
        )

        # Find nearest star
        nearest_idx = distances.idxmin()
        nearest_distance = distances[nearest_idx]

        if nearest_distance <= max_distance:
            star_data = self.catalog_df.loc[nearest_idx].to_dict()
            star_data['distance'] = nearest_distance
            return star_data

        return None

    def refresh(self):
        """Refresh the overlay (update catalog and redraw)."""
        self.update_catalog()
        if self.visible:
            self.draw()
