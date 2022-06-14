# Sudrabaino mākoņu sinhronās novērošanas apstrādes programmas

### Lietot bez Python pakotnes instalēšanas
```python -m pip install -r requirements.txt```

### Instalēt kā Python pakotni

```
python setup.py sdist
python -m pip install dist/sudrabainiemakoni-0.1.tar.gz
```
### Apstrādes piemērs ar komandlīnijas utilītu

Dots [sudrabaino mākoņu attēls](examples/TestCommandLine/js_202206120030.jpg) un tajā esošo [zvaigžņu pikseļu koordinātes failā](examples/TestCommandLine/js_202206120030_zvaigznes.txt).  Papildus kā komandlīnijas argumenti jāuzdod kameras ģeogrāfiskās koordinātes. Novērojuma datumu programma nolasa no attēla faila EXIF.  
Apstrāde notiek divos soļos: 
1. Tiek referencēts kameras skats un kameras parametri noglabāti JSON failos. Šajā solī arī tiek izveidots attēls ar ekvatoriālajām un horizontālajām koordinātēm. 
2. Sudrabaino mākoņu attēls tiek projicēts uz fiksēta augstumu virs zemes virsmas  
  
[Piemēra python fails](examples/TestCommandLine/testSM.py)  
[Komandlīnijas argumentu apraksts](examples/TestCommandLine/readme.md)


