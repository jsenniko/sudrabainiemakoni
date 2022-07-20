## Lietošanas piemēri 
Šajā katalogā atrodas lietošanas piemēri *Jupyter notebook* veidā

### [Referencēt kameru](ReferencetKameru.ipynb)
* Demonstrē kā inicializēt ```CloudImage``` tipa objektu, piesaistīt tam datumu, ģeogrāfiskās koordinātes un zvaigžņu sarakstu. 
* Kameras objekta kalibrācija, tā noglabāšana failā
* Noglabāt ```CloudImage``` tipa objektu failā
* Horizontālo un ekvatoriālo koordinātu līniju uzzīmēšana. Horizontālo koordināšu režģa uzzimēšana ļauj vizuāli pārliecināties par to, ka kamera ir referencēta pareizi attiecībā pret attēlu un zvaigznēm. Tāpat jāpievērš uzmanība kameras referencēšanas tolerances pikseļos, ko izvada ```cldim.PrepareCamera()``` funkcija
### [Projicēt](Projicet.ipynb)
* Ielasīt ```CloudImage``` tipa objektu no faila
* Inicializēt ```WebMercatorImage``` tipa objektu, kas ļauj veidot uz kartes projicētus sudrabaino mākoņu attēlus
* Noglabāt projicētos attēlus lietošanai GIS programmās
* Uzzīmēt projicētos attēlus uz OpenStreetMap pamatnes
### [Augstumu rindas](AugstumuRindas.ipynb)
* Doti divi vienlaicīgi sudrabaino mākoņu attēli no dažādām vietām
* Izveidot neatkarīgas projicēto attēlu rindas dažādos augstumos
* Izveidot dažādā stilā apvienotus projicētos attēlus dažādos augstumos. Aplūkojot izgatavotos failus, iespējams pēc mākoņu struktūru sakritības noteikt sudrabaino mākoņu augstumu
* Projicēt viena no pāra attēliem uz otra attēla kameras pozīciju mākoņiem atrodoties dažādos augstumos
