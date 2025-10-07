import os
import uuid
import argparse
def fileOk(s, ext):
        print('-----------------------')
        print(s)#, os.path.abspath(s))
        if os.path.isfile(s):
            if os.path.splitext(s)[1] in [ext]:
                return s
        raise argparse.ArgumentTypeError('Fails neeksistē vai ir nesavietojams!')
def newFileOk(s):
    dirname, filename = os.path.split(s)
    sdir = isDirectory(dirname)
    return s
def isDirectory(s):
    if os.path.isdir(os.path.abspath(s)):
        return s
    raise argparse.ArgumentTypeError('Katalogs neeksistē')
def isHeightRange(arg):
    try:
        hhlist = [float(s.strip()) for s in arg.split(",")]
        if len(hhlist) in [1,3]:
            return hhlist
    except:
        pass
    raise argparse.ArgumentTypeError('Jāuzdod augstums, vai no,līdz,solis')
def isHeightRangeOrHeightFile(arg):
    try:
        return isHeightRange(arg)
    except:
        return fileOk(arg,'.npy')

def parse_arguments(argumentlist=None):
    parser = argparse.ArgumentParser(description="Sudrabaino mākoņu attēlu referencēšana")
    parser.add_argument('--zvaigznes', type=str, action='store', help='Fails ar zvaigžņu pikseļu koordinātēm')
    parser.add_argument('--id', type=str, action='store', default=uuid.uuid4().hex, help='Identifikators')
    #TODO: validate lat,lon string
    #TODO: lat, lon from image exif
    parser.add_argument('--latlon', type=str, action='store', help='platums, garums')
    parser.add_argument('--plotRAGrid', type=isDirectory, action='store', help='RA koordinātu līniju attēlu katalogs')
    parser.add_argument('--plotAltAzGrid', type=isDirectory, action='store', help='AltAz koordinātu līniju attēlu katalogs')
    parser.add_argument('--loadCamera', type=str, action='store', help='No kurienes nolasīt kameras failu, ja neuzdod tad kalibrēt')
    parser.add_argument('--saveCamera', type=newFileOk, action='store', help='Kur noglabāt kameras failu')
    parser.add_argument('--webMercParameters', type=str, action='store', default='15,33,57,63,0.5', help='lonmin,lonmax,latmin,latmax,horizontal_resolution_km')
    parser.add_argument('--mapBounds', type=str, action='store', default='15,30,56,62', help='lonmin,lonmax,latmin,latmax')
    parser.add_argument('--mapAlpha', type=float, action='store', default=0.8, help='')
    parser.add_argument('--reprojectHeight', type=isHeightRange, action='store', default=80, help='Sudrabaino mākoņu augstums, km. Ja doti trīs ar komatu atdalīti skaitļi, tad augstums no,līdz,solis')
    parser.add_argument('--reprojectedMap', type=newFileOk, action='store', help='Georeferencēts attēls uz kartes fona')
    parser.add_argument('--reprojectedImage', type=newFileOk, action='store', help='Georeferencēts attēls jpg vai tif formātā bez kartes')
    parser.add_argument('--reprojectedImageFormat', type=str, choices=['jpg','tif'], action='store', help='Georeferencētā attēla formāts', default='jpg')
    parser.add_argument('--reprojectedImageJGW', action='store_true', help='Vai glabāt ģeoreferencētā attēla piesaistes failus prieķš GIS?')

    parser.add_argument('--loadProject', type=str, action='store', help='No kurienes nolasīt projekta failu')
    parser.add_argument('--saveProject', type=newFileOk, action='store', help='Kur noglabāt projekta failu')
    parser.add_argument('--optimizeDistortion',  action='store_true', help='Vai optimizēt kameras kropļojumus?')
    parser.add_argument('--notOptimizeCenter',   action='store_false',  help='Vai neoptimizēt optiskās ass centru?')
    parser.add_argument('--notOptimizeUnsymmetric',   action='store_false',  help='Vai neoptimizēt fokusa attālumus pa x un y atsevišķi?')

    parser.add_argument('file', action='store',  nargs=1, type=str, help='Attēla fails (jpg)')
    args=parser.parse_args(argumentlist)
    args.file=args.file[0]
    return args

def parse_arguments_timelapse(argumentlist=None):
    parser = argparse.ArgumentParser(description="Sudrabaino mākoņu laiklēciens")
    parser.add_argument('--loadProject', type=str, action='store', help='No kurienes nolasīt projekta failu', required=True)
    parser.add_argument('--webMercParameters', type=str, action='store', default='15,33,57,63,0.5', help='lonmin,lonmax,latmin,latmax,horizontal_resolution_km', required=True)
    parser.add_argument('--mapBounds', type=str, action='store', default='15,30,56,62', help='lonmin,lonmax,latmin,latmax')
    parser.add_argument('--mapAlpha', type=float, action='store', default=0.8, help='')
    parser.add_argument('--reprojectHeight', type=float, action='store', default=80, help='Sudrabaino mākoņu augstums, km', required=True)
    parser.add_argument('--timelapseInputDir', type=isDirectory, action='store', help='Laiklēciena attēlu katalogs', required=True)
    parser.add_argument('--timelapseInputFile', type=str, action='store', help='Laiklēciena attēlu saraksta fails')
    parser.add_argument('--timelapseOutputDir', type=newFileOk, action='store', help='Projicētā attēlu katalogs', required=True)
    parser.add_argument('--doNotPrepareGeoreferenced',   action='store_false',  help='', dest='prepareGeoreferenced')
    parser.add_argument('--prepareMaps',   action='store_true',  help='')
    args=parser.parse_args(argumentlist)
    return args
def parse_arguments_heightmap(argumentlist=None):
    parser = argparse.ArgumentParser(description="Sudrabaino mākoņu augstuma karte")
    parser.add_argument('--loadProject1', type=str, action='store', help='Pirmais projekta fails', required=True)
    parser.add_argument('--loadProject2', type=str, action='store', help='Otrais projekta fails', required=True)
    parser.add_argument('--loadControlPoints', type=str, action='store', help='Kontrolpunktu fails', required=True)
    parser.add_argument('--saveControlPoints', type=newFileOk, action='store', help='Kur noglabāt augstumus kontrolpunktos?')
    parser.add_argument('--plotControlPoints',  type=newFileOk, action='store', help='Kontrolpunktu attēlu faila šablons')
    parser.add_argument('--webMercParameters', type=str,
                        action='store', default='15,33,57,63,0.5', help='lonmin,lonmax,latmin,latmax,horizontal_resolution_km', required=True)
    parser.add_argument('--saveHeightMap', type=newFileOk, action='store', help='Kur noglabāt augstumu kartes bināro failu')
    parser.add_argument('--saveGeoMap', type=newFileOk, action='store', help='Kur noglabāt augstumu kartes attēlu')
    parser.add_argument('--mapBounds', type=str, action='store', default='15,30,56,62', help='lonmin,lonmax,latmin,latmax')
    args=parser.parse_args(argumentlist)
    return args
# def parse_arguments_reproject(argumentlist=None):
#     parser = argparse.ArgumentParser(description="Sudrabaino mākoņu projicēšana")
#     parser.add_argument('--loadProject', type=str, action='store', help='No kurienes nolasīt projekta failu', required=True)
#     parser.add_argument('--webMercParameters', type=str, action='store', default='15,33,57,63,0.5', help='lonmin,lonmax,latmin,latmax,horizontal_resolution_km', required=True)
#     parser.add_argument('--mapBounds', type=str, action='store', default='15,30,56,62', help='lonmin,lonmax,latmin,latmax')
#     parser.add_argument('--mapAlpha', type=float, action='store', default=0.8, help='')
#     parser.add_argument('--reprojectHeight', type=isHeightRangeOrHeightFile, action='store', default=80, help='Sudrabaino mākoņu augstums, km', required=True)
#     parser.add_argument('--doNotPrepareGeoreferenced',   action='store_false',  help='', dest='prepareGeoreferenced')
#     args=parser.parse_args(argumentlist)
#     return args
#"argString = '--outputFi# le=test.111'\n",
#"args = parser.parse_args(shlex.split(argString))"