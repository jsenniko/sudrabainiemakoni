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
    parser.add_argument('--reprojectHeight', type=float, action='store', default=80, help='Sudrabaino mākoņu augstums, km')
    parser.add_argument('--reprojectedMap', type=newFileOk, action='store', help='Georeferencēts attēls')

    parser.add_argument('file', action='store',  nargs=1, type=str, help='Attēla fails (jpg)')
    args=parser.parse_args(argumentlist)
    args.file=args.file[0]
    return args
#"argString = '--outputFile=test.111'\n",
#"args = parser.parse_args(shlex.split(argString))"