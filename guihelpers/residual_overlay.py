"""
ResidualOverlay - Manages visualization of camera calibration residuals.
This class displays residuals as scatter points (magnitude) and quiver arrows (vectors).
"""

import numpy as np


class ResidualOverlay:
    """
    Manages the display of camera calibration residuals as a matplotlib overlay.

    Shows two types of visualizations:
    1. Scatter plot: Magnitude of residuals at star positions
    2. Quiver plot: Vectors from model prediction to actual star position
    """

    def __init__(self, ax):
        """
        Initialize the residual overlay.

        Args:
            ax: matplotlib axes object
        """
        self.ax = ax

        # Residual data
        self.star_pixel_coords = None  # Actual digitized star positions (N, 2)
        self.model_pixel_coords = None  # Model-predicted star positions (N, 2)
        self.residuals = None  # model - actual (N, 2)
        self.residual_magnitudes = None  # sqrt(dx^2 + dy^2) (N,)

        # Visualization artists
        self.scatter_artist = None
        self.quiver_artist = None

        # Visibility flags
        self.scatter_visible = False
        self.quiver_visible = False

    def set_residuals(self, star_pixel_coords, model_pixel_coords):
        """
        Update residual data.

        Args:
            star_pixel_coords: Actual digitized star positions, shape (N, 2)
            model_pixel_coords: Model-predicted star positions, shape (N, 2)
        """
        self.star_pixel_coords = np.array(star_pixel_coords)
        self.model_pixel_coords = np.array(model_pixel_coords)

        # Calculate residuals (model - actual)
        self.residuals = self.model_pixel_coords - self.star_pixel_coords

        # Calculate magnitudes
        self.residual_magnitudes = np.sqrt(np.sum(self.residuals**2, axis=1))

        # Redraw if visible
        if self.scatter_visible or self.quiver_visible:
            self.draw()

    def show_scatter(self):
        """Show scatter plot of residual magnitudes."""
        if not self.scatter_visible:
            self.scatter_visible = True
            self.draw()

    def hide_scatter(self):
        """Hide scatter plot of residual magnitudes."""
        if self.scatter_visible:
            self.scatter_visible = False
            self._remove_scatter()
            self.ax.figure.canvas.draw_idle()

    def show_quiver(self):
        """Show quiver plot of residual vectors."""
        if not self.quiver_visible:
            self.quiver_visible = True
            self.draw()

    def hide_quiver(self):
        """Hide quiver plot of residual vectors."""
        if self.quiver_visible:
            self.quiver_visible = False
            self._remove_quiver()
            self.ax.figure.canvas.draw_idle()

    def toggle_scatter(self):
        """Toggle scatter plot visibility."""
        if self.scatter_visible:
            self.hide_scatter()
        else:
            self.show_scatter()

    def toggle_quiver(self):
        """Toggle quiver plot visibility."""
        if self.quiver_visible:
            self.hide_quiver()
        else:
            self.show_quiver()

    def draw(self):
        """Draw all visible residual visualizations."""
        if self.residuals is None or len(self.residuals) == 0:
            print("No residual data to display")
            return

        # Remove old artists
        self._remove_all()

        # Draw scatter if visible
        if self.scatter_visible:
            self._draw_scatter()

        # Draw quiver if visible
        if self.quiver_visible:
            self._draw_quiver()

        self.ax.figure.canvas.draw_idle()

    def _draw_scatter(self):
        """Draw scatter plot of residual magnitudes."""
        if self.residual_magnitudes is None:
            return

        # Use actual star positions for scatter plot locations
        x = self.star_pixel_coords[:, 0]
        y = self.star_pixel_coords[:, 1]

        # Color-code by residual magnitude
        # Use a color map: blue (small) to red (large)
        self.scatter_artist = self.ax.scatter(
            x, y,
            c=self.residual_magnitudes,
            s=100,  # Size of markers
            cmap='coolwarm',
            alpha=0.7,
            edgecolors='black',
            linewidths=1,
            zorder=1000,  # Draw on top
            label=f'Residuals (RMS: {np.sqrt(np.mean(self.residual_magnitudes**2)):.2f}px)'
        )

        # Add colorbar if not already present
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            # Create colorbar as an inset to avoid interference with zoom/pan
            # Place it in the upper right corner of the axes
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(
                self.ax,
                width="3%",  # Width of colorbar
                height="30%",  # Height of colorbar
                loc='upper right',
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=self.ax.transAxes,
                borderpad=1
            )
            self.colorbar = self.ax.figure.colorbar(
                self.scatter_artist,
                cax=cax,
                label='Residual (px)'
            )

    def _draw_quiver(self):
        """Draw quiver plot showing residual vectors."""
        if self.residuals is None:
            return

        # Quiver starts at actual star position and points toward model position
        x = self.star_pixel_coords[:, 0]
        y = self.star_pixel_coords[:, 1]
        u = self.residuals[:, 0]  # dx (model - actual)
        v = self.residuals[:, 1]  # dy (model - actual)

        # Draw arrows from actual to model position
        self.quiver_artist = self.ax.quiver(
            x, y, u, v,
            angles='xy',
            scale_units='xy',
            scale=1,  # 1:1 scale (1 pixel = 1 pixel)
            color='red',
            alpha=0.8,
            width=0.003,
            headwidth=5,
            headlength=7,
            zorder=1001,  # Draw on top of scatter
            label='Residual vectors'
        )

    def _remove_scatter(self):
        """Remove scatter plot artist."""
        if self.scatter_artist is not None:
            try:
                self.scatter_artist.remove()
            except (ValueError, KeyError):
                # Artist already removed (e.g., by figure clear)
                pass
            self.scatter_artist = None

        # Remove colorbar
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except (ValueError, KeyError, AttributeError):
                # Colorbar axes already removed (e.g., by figure clear)
                # or colorbar is in an invalid state
                pass
            self.colorbar = None

    def _remove_quiver(self):
        """Remove quiver plot artist."""
        if self.quiver_artist is not None:
            try:
                self.quiver_artist.remove()
            except (ValueError, KeyError):
                # Artist already removed (e.g., by figure clear)
                pass
            self.quiver_artist = None

    def _remove_all(self):
        """Remove all artists."""
        self._remove_scatter()
        self._remove_quiver()

    def clear(self):
        """Clear all residual data and visualizations."""
        self._remove_all()
        self.star_pixel_coords = None
        self.model_pixel_coords = None
        self.residuals = None
        self.residual_magnitudes = None
        self.ax.figure.canvas.draw_idle()

    def refresh(self):
        """Refresh the overlay (redraw if visible)."""
        if self.scatter_visible or self.quiver_visible:
            self.draw()
