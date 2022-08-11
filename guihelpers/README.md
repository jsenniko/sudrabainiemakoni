## Lietotāja interfeiss sudrabaino mākoņu attēlu apstrādei
### Palaišana
Jāpāriet uz katalogu ```guihelpers``` tad var palaist
```
python sudrabainie.py
```
### Darbību secība
1. Ciparot zvaigznes.
    1. Ielasīt attēlu. Ielasīšanas gaitā vajadzēs ievadīt informāciju par platumu un garumu (```platums,garums```). Pēc ievades programma izveidos platuma un garuma failu tajā pašā katalogā, kur ir attēla fails. Ja platuma un garuma fails būs izveidots, tad ievadīt neprasīs.
    2. Ciparot zvaigznes. Uz attēla ar kreiso peles taustiņu atzīmēt zvaigzni, programma prasīs ievadīt zvaigznes nosaukumu. Teksta logā var sekot līdzi, vai zvaigzne atpazīta. Atpazīšanai nepieciešams interneta pieslēgums. Zvaigznes nosakuma vietā var ievadīt ar komatu atdalītus rektascensiju un deklināciju grādos. Beigt ciparot var nospiežot labo peles taustiņu, vai vēlreiz izvēloties izvēlni *ciparot zvaigznes*. Beidzot ciparot programma noglabās zvaigžņu failu tajā pašā katalogā, kur attēls, pajautājot. Pietuvināt var lietojot grafiskā loga pogas. Ar iespiestām grafiskā loga pogām ciparot nevar, pēc pietuvināšanas tās jāatslēdz, lai turpinātu ciparot.
  3. Kalibrēt kameru. Nospiežot izvēlni teksta logā parādīsies kalibrācijas statistika un kalibrētās kameras parametri.
  4. Saglabāt projektu. Projekta fails papildus norādei uz attēlu satur kalibrētās kameras parametrus. Turpmākajām darbībām nepieciešams to saglabāt.
  5. Ielasīt projektu. Ja projekts saglabāts, tad var to nolasīt, izlaižot darbības 1.-4.
  6. Zīmēt horizontālo koordinātu režģi. Nepieciešams, lai vizuāli pārliecinātos par kameras kalibrācijas labumu.
  7. Projicēt. Lietotājam jāievada provizoriskais mākoņu augstums kilometros, programma uzzīmēs uz kartes telpiski referencētu attēlu
  8. Ielasīt otro projektu. Nepieciešams, lai veiktu sinhronās novērošanas analīzi. Otro projektu pirms tam jāsagatavo - jāciparo zvaigznes, jākalibrē kamera (sk. punktus 1.-4.).  
  9. Ciparot kontrolpunktus
  10. Zīmēt kotrolpunktu augstumus
  11. Izveidot augstumu karti
  12. Projicēt attēlus, izmantojot augstumu karti
  13. Saglabāt projicētos attēlus JPG un TIFF formātā. Papildus tiks saglabāti JGW un TFW koordināšu piesaistes faili, tos kopā ar projicētajiem attēliem var ielasīt kā GIS rastrus (GIS programmatūrā norādos koordinātu sistēmu [EPSG:3857](https://epsg.io/3857)). Noglabājot TIF failu, papildus tiks saglabāts arī KML fails ielasīšanai Google Earth Pro.

### Izvēlnes
- Fails
  - Ielasīt attēlu
  - Ielasīt projektu
  - Saglabāt projektu
  - Ielasīt otro projektu
  - Ielasīt kontrolpunktus
  - Ielasīt augstumu karti
  - Saglabāt augstumu karti
  - Saglabāt projicēto attēlu JPG
  - Saglabāt projicēto attēlu TIF
- Darbības
  -  Ciparot zvaigznes
  -  Kalibrēt kameru
  -  Kontrolpunkti
  -  Izveidot augstumu karti
  -  Projekcijas apgabals
  -  Kartes apgabals
  -  Kameras kalibrācijas parametri
- Zīmēt
  - Attēls
  - Horizontālo koordinātu režģis
  - Projicēt
  - Projicēt kopā
  - Projicēt no augstumu kartes
  - Projicēt no augstumu kartes kopā
  - Kontrolpunktu augstumus
  - Augstumu karti
