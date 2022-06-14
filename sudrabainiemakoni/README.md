## Sudrabaino mākoņu apstrādes pakotne  
Pakotnes pamatdaļa atrodas failā ```cloudimage.py```. Pamata objekts ir ```CloudImage``` - tas atbilst vienam sudrabaino mākoņu attēlam.
Soļi objekta inicializācijai ir sekojoši:
Inicializējam norādot identifikatoru un attēla jpg faila nosaukumu
```
cldim = CloudImage('js_202206120030', 'js_202206120030.jpg')
```
Norādam datumu, pieprasot to no EXIF
```
cldim.setDateFromExif()
```
Alternatīvi varam arī tieši uzstādīt datumu
```
cldim.setDate(datetime.datetime(2021,6,15,0,30,21))
```
Uzstādām novērotāja ģeogrāfisko platumu un garumu (grādos)
```
cldim.setLocation(lat=lat, lon=lon)
```
Uzstādām zvaigžņu nosakumu sarakstu un pikseļu koordinātu sarakstu. Nosaukumiem jāatbilst tādiem zvaigžņu nosaukumiem, ko atpazīst [SIMBAD](http://cds.u-strasbg.fr/cgi-bin/Sesame) astronomisko objektu pieprasījumu datubāze Pikseļu koordinātes ir saraksts [[ix1,iy1],[ix2,iy2],...]. Zvaigžņu nosaukumu un pikseļu koordināšu saraksta garumam jābūt vienādam.
```
cldim.setStarReferences(starnames, pixels)
```
Ja nepieciešams uzzīmēt ekvatoriālo koordinātu režģi uz attēla, tad vispirs jāiegūst attēlam piesaistīta WCS koordinātu sistēma
```
cldim.GetWCS(sip_degree=2, fit_parameters={'projection':'TAN'})
```
Tad ekvatoriālo koordinātu režģi uz attēla iegūst sekojoši, norādot katalogu, kurā noglabāt attēlu. Attēla vārds būs formā ekv_koord_{id}.jpg
```
plots.PlotRADecGrid(cldim, outImageDir = katalogs,  stars = False, showplot=False )
```
Pēc kameras referencēšanas, var iegūt attēlu ar horizontālo koordinātu režģi, norādot katalogu, kurā noglabāt attēlu. Attēla vārds būs formā horiz_koord_{id}.jpg
```
plots.PlotAltAzGrid(cldim,  outImageDir = katalogs, stars = True, showplot=False, from_camera = True)
```

