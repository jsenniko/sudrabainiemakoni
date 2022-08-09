## Lietotāja interfeiss sudrabaino mākoņu attēlu apstrādei
### Palaišana
```
python sudrabainie.py
```
### Darbību secība
1. Ciparot zvaigznes.
    1. Ielasīt attēlu. Ielasīšanas gaitā vajadzēs ievadīt informāciju par platumu un garumu (```platums,garums```). Pēc ievades programma izveidos platuma un garuma failu tajā pašā katalogā, kur ir attēla fails. Ja platuma un garuma fails būs izveidots, tad ievadīt neprasīs.
    2. Ciparot zvaigznes. Uz attēla ar kreiso peles taustiņu atzīmēt zvaigzni, programma prasīs ievadīt zvaigznes nosaukumu. Teksta logā var sekot līdzi, vai zvaigzne atpazīta. Beigt ciparot var nospiežot labo peles taustiņu, vai vēlreiz izvēloties izvēlni *ciparot zvaigznes*. Beizot ciparot programma noglabās zvaigžņu failu tajā pašā katalogā, kur attēls. Pietuvināt var lietojot grafiskā loga pogas. 
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

### Izvēlnes
- Fails
  - Ielasīt attēlu
  - Ielasīt projektu
  - Saglabāt projektu
  - Ielasīt otro projektu
  - Ielasīt kontrolpunktus
  - Ielasīt augstumu karti
  - Saglabāt augstumu karti
- Darbības
  -  Ciparot zvaigznes
  -  Kalibrēt kameru
  -  Projekcijas apgabals
  -  Kartes apgabals
  -  Kontrolpunkti
  -  Izveidot augstumu karti
- Zīmēt
  - Attēls
  - Horizontālo koordinātu režģis
  - Projicēt
  - Projicēt kopā
  - Projicēt no augstumu kartes
  - Projicēt no augstumu kartes kopā
  - Kontrolpunktu augstumus
  - Augstumu karti
