# Sudrabaino mākoņu sinhronās novērošanas apstrādes programmas

### Pakotnes lietošanas instrukcija
[sudrabainiemakoni/README.md](sudrabainiemakoni/README.md)

### Instalācija
#### Lietošanai bez Python pakotnes instalēšanas

Ar pip utilītu instalējam vajadzīgās pakotnes  
```python -m pip install -r requirements.txt```   
Apakšprogrammas atrodas katalogā [**sudrabainiemakoni**](./sudrabainiemakoni). Veidojot python skriptu, lai lietotu pakotni, jāpievieno repozitorija katalogs pie python ceļiem 
```
import sys
sys.path.append(cels_uz_repositorija_katalogu)
from sudrabainiemakoni.cloudimage import CloudImage
```

#### Instalējot kā Python pakotni

Izveidojam pakotni
```
python setup.py sdist
```
Instalējam pakotni
```
python -m pip install dist/sudrabainiemakoni-0.1.tar.gz
```
Tālāk var tieši importēt vajadzīgās komponentes
```
from sudrabainiemakoni.cloudimage import CloudImage
```

### Apstrādes piemērs ar komandlīnijas utilītu

Dots [sudrabaino mākoņu attēls](examples/TestCommandLine/js_202206120030.jpg) un tajā esošo [zvaigžņu pikseļu koordinātes failā](examples/TestCommandLine/js_202206120030_zvaigznes.txt).  Papildus kā komandlīnijas argumenti jāuzdod kameras ģeogrāfiskās koordinātes. Novērojuma datumu programma nolasa no attēla faila EXIF.  
Apstrāde notiek divos soļos: 
1. Tiek referencēts kameras skats un kameras parametri noglabāti JSON failos. Šajā solī arī tiek izveidots attēls ar ekvatoriālajām un horizontālajām koordinātēm. 
2. Sudrabaino mākoņu attēls tiek projicēts uz fiksēta augstumu virs zemes virsmas  
  
[Piemēra python fails](examples/TestCommandLine/testSM.py)  
[Komandlīnijas argumentu apraksts](examples/TestCommandLine/readme.md)


