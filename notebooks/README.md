## Lietošanas piemēri 
Šajā katalogā atrodas lietošanas piemēri *Jupyter notebook* veidā

### [ReferencetKameru](ReferencetKameru.ipynb)
* Demonstrē kā inicializēt ```CloudImage``` tipa objektu, piesaistīt tam datumu, ģeogrāfiskās koordinātes un zvaigžņu sarakstu. 
* Kameras objekta kalibrācija, tā noglabāšana failā
* Horizontālo un ekvatoriālo koordinātu līniju uzzīmēšana. Horizontālo koordināšu režģa uzzimēšana ļauj vizuāli pārliecināties par to, ka kamera ir referencēta pareizi attiecībā pret attēlu un zvaigznēm. Tāpat jāpievērš uzmanība kameras referencēšanas tolerances pikseļos, ko izvada ```cldim.PrepareCamera()``` funkcija
