#!/usr/bin/env python3
"""
One Image Workflow Example
Implements the workflow described in sudrabainiemakoni/README.md
"""

import os
import pandas as pd
import numpy as np
import datetime

# Add parent directory to path to import sudrabainiemakoni
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sudrabainiemakoni.cloudimage import CloudImage, WebMercatorImage
from sudrabainiemakoni import plots
from sudrabainiemakoni.wcs_coordinate_systems import WCSCoordinateSystemsAdapter

def main():
    """Main workflow function implementing README.md steps"""
    
    # Configuration
    image_id = 'js_202106210100'
    sample_data_dir = 'examples/SampleData'
    output_dir = 'output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Initialize CloudImage
    print("Step 1: Initializing CloudImage...")
    image_file = os.path.join(sample_data_dir, f'{image_id}.jpg')
    cldim = CloudImage(image_id, image_file)
    
    # Step 2: Set date from EXIF
    print("Step 2: Setting date from EXIF...")
    cldim.setDateFromExif()
    print(f"Date from EXIF: {cldim.date}")
    
    # Step 3: Set observer location (example coordinates for Latvia)
    print("Step 3: Setting observer location...")
    lat, lon = 56.693, 23.656  
    cldim.setLocation(lat=lat, lon=lon)
    print(f"Location set to: {lat}°N, {lon}°E")
    
    # Step 4: Load star references from file
    print("Step 4: Loading star references...")
    star_file = os.path.join(sample_data_dir, f'{image_id}_zvaigznes.txt')
    df = pd.read_csv(star_file, sep='\t', header=None)
    starnames = df[0].tolist()
    pixels = np.array(df[[1, 2]])
    cldim.setStarReferences(starnames, pixels)
    print(f"Loaded {len(starnames)} star references")
    
    # Step 5: Create WCS coordinate system (using adapter for refactored code)
    print("Step 5: Creating WCS coordinate system...")
    wcs_adapter = WCSCoordinateSystemsAdapter(cldim)
    wcs = wcs_adapter.GetWCS(sip_degree=2, fit_parameters={'projection': 'TAN'})
    print("WCS created successfully")
    
    # Step 6: Plot equatorial coordinate grid
    print("Step 6: Plotting equatorial coordinate grid...")
    plots.PlotRADecGrid(cldim, outImageDir=output_dir, stars=False, showplot=False)
    print(f"Equatorial grid saved as ekv_coord_{image_id}.jpg")
    
    # Step 7: Prepare camera referencing
    print("Step 7: Preparing camera referencing...")
    cldim.PrepareCamera()
    print("Camera prepared successfully")
    
    # Step 8: Save camera parameters
    print("Step 8: Saving camera parameters...")
    camera_file = os.path.join(output_dir, f'{image_id}_camera')
    cldim.SaveCamera(camera_file)
    print(f"Camera saved to {camera_file}_*.json files")
    
    # Step 9: Plot horizontal coordinate grid
    print("Step 9: Plotting horizontal coordinate grid...")
    plots.PlotAltAzGrid(cldim, outImageDir=output_dir, stars=True, showplot=False, from_camera=True)
    print(f"Horizontal grid saved as horiz_coord_{image_id}_zvaigznes.jpg")
    
    # Step 10: Create WebMercator projection
    print("Step 10: Creating WebMercator projection...")
    # Define map bounds (Baltic region)
    lonmin, lonmax = 15.0, 35.0
    latmin, latmax = 57.0, 64.0
    horizontal_resolution_km = 0.5
    
    webmerc = WebMercatorImage(cldim, lonmin, lonmax, latmin, latmax, horizontal_resolution_km)
    
    # Step 11: Reproject to specific height
    print("Step 11: Reprojecting to 80km height...")
    reproject_height = 80  # km
    webmerc.prepare_reproject_from_camera(reproject_height)
    projected_image_hght = webmerc.Fill_projectedImageMasked()
    print("Reprojection completed")
    
    # Step 12: Create georeferenced map
    print("Step 12: Creating georeferenced map...")
    camera_points = [[cldim.location.lon.value, cldim.location.lat.value]]
    map_filename = os.path.join(output_dir, f'map_{reproject_height}_{image_id}.jpg')
    
    # Map bounds for visualization
    map_lonmin, map_lonmax = 15.0, 35.0
    map_latmin, map_latmax = 56.0, 64.0
    
    plots.PlotReferencedImages(webmerc, [projected_image_hght],
                             camera_points=camera_points,
                             outputFileName=map_filename,
                             lonmin=map_lonmin, lonmax=map_lonmax, 
                             latmin=map_latmin, latmax=map_latmax,
                             alpha=0.95)
    print(f"Georeferenced map saved as {map_filename}")
    
    print("\nWorkflow completed successfully!")
    print(f"Output files saved in: {output_dir}/")

if __name__ == "__main__":
    main()