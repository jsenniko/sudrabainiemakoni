# One Image Workflow Example

This example implements the complete workflow described in `sudrabainiemakoni/README.md` for processing a single noctilucent cloud image.

## Usage

1. Ensure you have the required dependencies installed
2. Run the workflow:

```bash
cd examples/OneImageWorkflow
python workflow.py
```

## Workflow Steps

The script implements the following steps from the main README:

1. **Initialize CloudImage** - Create CloudImage object with image ID and file
2. **Set Date** - Extract date from EXIF data
3. **Set Location** - Define observer's geographic coordinates
4. **Load Star References** - Read star names and pixel coordinates from file
5. **Create WCS** - Generate World Coordinate System using star references
6. **Plot Equatorial Grid** - Generate RA/Dec coordinate grid overlay
7. **Prepare Camera** - Calculate camera positioning in topocentric coordinates
8. **Save Camera** - Export camera parameters to JSON files
9. **Plot Horizontal Grid** - Generate Alt/Az coordinate grid overlay
10. **Create WebMercator Projection** - Set up map projection parameters
11. **Reproject Image** - Project image to fixed altitude (80km)
12. **Generate Georeferenced Map** - Create final map with OpenStreetMap base

## Input Files

- `../SampleData/js_202206120030.jpg` - Noctilucent cloud image
- `../SampleData/js_202206120030_zvaigznes.txt` - Star reference coordinates

## Output Files

All outputs are saved to the `output/` directory:

- `ekv_coord_js_202206120030.jpg` - Image with equatorial coordinate grid
- `horiz_coord_js_202206120030_zvaigznes.jpg` - Image with horizontal coordinate grid and stars
- `js_202206120030_camera_enu.json` - Camera parameters in ENU coordinates
- `js_202206120030_camera_ecef.json` - Camera parameters in ECEF coordinates
- `map_80_js_202206120030.jpg` - Final georeferenced map at 80km altitude

## Technical Notes

- Uses the refactored `WCSCoordinateSystemsAdapter` for WCS functionality
- Observer location set to Riga, Latvia coordinates as example
- Map projection covers Baltic region (20°E-30°E, 54°N-60°N)
- Noctilucent clouds projected to 80km altitude (typical height)
- Camera referencing provides pixel-level accuracy for geometric calculations