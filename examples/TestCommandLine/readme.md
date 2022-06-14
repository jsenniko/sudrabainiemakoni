usage: testSM.py [-h] [--zvaigznes ZVAIGZNES] [--id ID] [--latlon LATLON]
                 [--plotRAGrid PLOTRAGRID] [--plotAltAzGrid PLOTALTAZGRID]
                 [--loadCamera LOADCAMERA] [--saveCamera SAVECAMERA]
                 [--webMercParameters WEBMERCPARAMETERS]
                 [--mapBounds MAPBOUNDS] [--mapAlpha MAPALPHA]
                 [--reprojectHeight REPROJECTHEIGHT]
                 [--reprojectedMap REPROJECTEDMAP]
                 file

Sudrabaino mākoņu attēlu referencēšana

positional arguments:
  file                  Attēla fails (jpg)

optional arguments:
  -h, --help            show this help message and exit
  --zvaigznes ZVAIGZNES
                        Fails ar zvaigžņu pikseļu koordinātēm
  --id ID               Identifikators
  --latlon LATLON       platums, garums
  --plotRAGrid PLOTRAGRID
                        RA koordinātu līniju attēlu katalogs
  --plotAltAzGrid PLOTALTAZGRID
                        AltAz koordinātu līniju attēlu katalogs
  --loadCamera LOADCAMERA
                        No kurienes nolasīt kameras failu, ja neuzdod tad
                        kalibrēt
  --saveCamera SAVECAMERA
                        Kur noglabāt kameras failu
  --webMercParameters WEBMERCPARAMETERS
                        lonmin,lonmax,latmin,latmax,horizontal_resolution_km
  --mapBounds MAPBOUNDS
                        lonmin,lonmax,latmin,latmax
  --mapAlpha MAPALPHA
  --reprojectHeight REPROJECTHEIGHT
                        Sudrabaino mākoņu augstums, km
  --reprojectedMap REPROJECTEDMAP
                        Georeferencēts attēls