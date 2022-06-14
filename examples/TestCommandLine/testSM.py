import os
import sys
import shlex
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from sudrabainiemakoni import procesetSM, argumentsSM

id='js_202206120030'
latlon='56.655488,23.734581'
height=80

#argString0=f"""-h"""
#arglist = shlex.split(argString0)
#args = argumentsSM.parse_arguments(arglist)

argString1=f"""--id={id}
--zvaigznes={id}_zvaigznes.txt
--latlon={latlon}
--plotRAGrid=
--plotAltAzGrid=
--saveCamera=kam_{id}.json {id}.jpg"""
arglist = shlex.split(argString1)
args = argumentsSM.parse_arguments(arglist)
print(args)
procesetSM.doProcessing(args)

argString2=f"""--id={id}
--latlon={latlon}
--loadCamera=kam_{id}.json
--mapBounds=15,35,56,64
--webMercParameters=15,35,57,64,0.5
--mapAlpha=0.95
--reprojectHeight={height}
--reprojectedMap=map_{height}_{id}.jpg {id}.jpg"""

arglist = shlex.split(argString2)
args = argumentsSM.parse_arguments(arglist)
print(args)
procesetSM.doProcessing(args)