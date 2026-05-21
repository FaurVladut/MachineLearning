Model pentru prezicerea bolii de inima.

Am decis sa fac un model de prezicere a sansei ca un pacient sa aiba o boala de inima datorita utilitatii acestor modele in situatii din viata reala.
Organizația mondială a sănătății  estimează că 17,9 milioane de oameni mor în fiecare an din cauza bolilor cardiovascular.
De asemenea am decis sa fac aceasta aplicatie datorita faptului ca este un bun prim pas pentru a invata conceptul de machine learning.
Tehnologii folosite: jupyter notebook si python. 

Database-ul care va fi folosit pentru acest model se afla la: https://archive.ics.uci.edu/dataset/45/heart+disease 
Acest dabase are datele a 303 pacienti incluzand urmatoarele atribute:

Descrierea Setului de Date

Setul de date conține următoarele caracteristici (atribute):

    Age: vârsta pacientului [ani]

    Sex: sexul pacientului [M: Masculin, F: Feminin]

    ChestPainType: tipul durerii în piept [TA: Angină Tipică, ATA: Angină Atipică, NAP: Durere Non-Anginoasă, ASY: Asimptomatic]

    RestingBP: tensiunea arterială în repaus [mm Hg]

    Cholesterol: colesterolul seric [mm/dl]

    FastingBS: glicemia pe nemâncate [1: dacă FastingBS > 120 mg/dl, 0: altfel]

    RestingECG: rezultatele electrocardiogramei în repaus [Normal: Normal, ST: prezintă anomalii ale undei ST-T (inversări ale undei T și/sau supradenivelare sau subdenivelare ST > 0.05 mV), LVH: arată o hipertrofie ventriculară stângă probabilă sau certă conform criteriilor Estes]

    MaxHR: frecvența cardiacă maximă atinsă [valoare numerică între 60 și 202]

    ExerciseAngina: angină indusă de efort [Y: Da, N: Nu]


1. Introducere și Motivație

Contextul general: Bolile cardiovasculare sunt principala cauză de mortalitate la nivel global. Conform Organizației Mondiale a Sănătății (OMS), se estimează că 17.9 milioane de oameni mor anual din cauza acestor afecțiuni.
Utilitatea practică: Un model de Machine Learning capabil să prezică riscul de boală cardiacă pe baza unor analize de rutină (tensiune, colesterol, EKG) poate acționa ca un sistem de suport pentru deciziile medicilor (Clinical Decision Support System), ajutând la trierea rapidă a pacienților cu risc ridicat.
Motivația personală: Am ales această problemă deoarece reprezintă un caz clasic, dar vital, de clasificare binară, fiind un punct de plecare excelent pentru a înțelege întregul flux de lucru într-un proiect de Machine Learning (de la curățarea datelor până la explicabilitatea modelului).
2. Descrierea Datelor și Contextul Proiectului
Setul de date: Proiectul utilizează setul de date Heart Disease (Cleveland) din repository-ul UCI Machine Learning.
Număr de instanțe: 297 pacienti.
Tipul învățării: Învățare supervizată - Clasificare binară (scopul este de a prezice clasa 0 = Sănătos, sau 1 = Bolnav).
Atributele folosite (aliniate cu codul):
age: vârsta pacientului (ani)
sex: sexul pacientului (1 = masculin, 0 = feminin)
cp: tipul durerii în piept (1: angină tipică, 2: angină atipică, 3: durere non-anginoasă, 4: asimptomatic)
trestbps: tensiunea arterială în repaus (mm Hg)
chol: colesterolul seric (mg/dl)
fbs: glicemia pe nemâncate > 120 mg/dl (1 = da, 0 = nu)
restecg: rezultate EKG în repaus (0, 1, 2)
thalach: ritmul cardiac maxim atins
exang: angină indusă de efort (1 = da, 0 = nu)
oldpeak: subdenivelarea segmentului ST indusă de efort
slope: panta segmentului ST la efort maxim (1, 2, 3)
ca: numărul de vase de sânge majore colorate prin fluoroscopie (0-3)
thal: defect (3 = normal, 6 = fix, 7 = reversibil)
target: variabila țintă (binarizată în cod: 0 = absența bolii, 1 = prezența bolii)
3. Aspecte Teoretice (State-of-the-Art)
3.1. Regresia Logistică și Funcția Sigmoidă
Deși poartă denumirea de „regresie”, Regresia Logistică este, în esență, un algoritm fundamental de Machine Learning utilizat pentru clasificarea binară (separarea datelor în exact două categorii, cum ar fi 0 = Sănătos și 1 = Bolnav). Spre deosebire de modelele care estimează valori continue (cum ar fi prezicerea prețului unei case), Regresia Logistică este antrenată să estimeze o probabilitate procentuală.
Cum funcționează (Intuiția algoritmului)
Modul în care acest algoritm ia decizii este foarte similar cu modul în care un medic evaluează un pacient, doar că o face la o scară matematică. Când analizează un pacient, modelul ia în considerare fiecare caracteristică în parte (vârsta, nivelul de colesterol, ritmul cardiac maxim etc.).
În timpul antrenamentului, algoritmul învață să asocieze o anumită „greutate” sau importanță fiecărui simptom. De exemplu, poate învăța că un tip de durere toracică specific (cp) atârnă foarte greu în decizia finală, în timp ce nivelul glicemiei (fbs) contează mai puțin. Pentru fiecare pacient nou, modelul adună toate aceste informații ponderate și obține un scor brut de risc.
Rolul esențial al Funcției Sigmoide
Problema cu scorul brut este că poate lua orice valoare (de exemplu, -150 sau +3000). În contextul medical, un scor de risc de „+3000” nu are sens practic; medicii au nevoie de o evaluare exprimată în procente.
Aici intervine funcția sigmoidă. Aceasta acționează ca un filtru sau o „pâlnie” cu o formă de „S”. Ea preia scorul brut, oricât de extrem ar fi, și îl „comprimă” forțat într-un interval strict între 0 și 1 (adică între 0% și 100%).
Dacă scorul brut al pacientului este pozitiv și foarte mare, funcția sigmoidă îl împinge aproape de 1 (probabilitate maximă de a avea boala).
Dacă scorul brut este foarte mic sau negativ, funcția îl apropie de 0.
Odată ce avem această probabilitate (de exemplu, 0.85, adică 85%), decizia algoritmului este simplă: se stabilește un prag de decizie (de obicei 50%). Dacă probabilitatea depășește pragul, pacientul este etichetat cu clasa 1 (Bolnav), altfel, cu clasa 0 (Sănătos).
De ce este optimă Regresia Logistică pentru domeniul medical?
Deși astăzi există algoritmi mult mai sofisticați, Regresia Logistică rămâne un standard de aur în cercetarea clinică din următoarele motive:
Explicabilitatea totală (White-box model): Algoritmul este complet transparent. Putem inspecta direct cât de mult a cântărit fiecare analiză în decizia finală. Această transparență oferă încredere medicilor, care pot valida deciziile algoritmului cu literatura medicală existentă.
Oferă un context prin probabilități: Sistemul nu se limitează la un simplu „Da” sau „Nu”. Faptul că modelul poate spune „Pacientul A are 95% șanse de boală, iar Pacientul B are 52%” schimbă total modul în care acționează un medic la triaj, deși ambii pacienți au primit eticheta de „Bolnav”.
Stabilitatea pe date puține: Modelul caută cea mai simplă graniță de decizie între pacienții bolnavi și cei sănătoși. Această simplitate îl împiedică să „învețe pe de rost” zgomotul din date (fenomenul de overfitting), fiind extrem de robust pentru seturi de date de dimensiuni reduse, cum este setul Cleveland (care are doar 303 înregistrări).
3.2. Random Forest: Învățarea de tip Ensemble, Bagging și Arborii de Decizie
Random Forest (Pădurea Aleatoare) este unul dintre cei mai populari și robuști algoritmi de Machine Learning. Pentru a înțelege cum funcționează, trebuie să pornim de la baza sa constructivă — arborele de decizie — și să înțelegem cum algoritmul combină sute de astfel de arbori folosind tehnica de „învățare în ansamblu” (Ensemble Learning).
Arborii de Decizie: Piesele de bază
Un Arbore de Decizie (Decision Tree) funcționează exact ca o schemă logică (flowchart) pe care ar urma-o un medic la camera de gardă. Când evaluează un pacient, arborele pune o serie de întrebări secvențiale, bazate pe date. De exemplu:
„Vârsta pacientului este mai mare de 55 de ani?” (Dacă Da, mergi pe ramura din dreapta; dacă Nu, pe cea din stânga).
„Tipul de durere în piept (cp) este de nivel 4 (asimptomatic)?”
„Ritmul cardiac maxim (thalach) este sub 140?”
La finalul acestor ramificări (ramuri), se află „frunzele” arborelui, care reprezintă decizia finală: Sănătos sau Bolnav. Deși sunt foarte intuitivi, arborii de decizie singuri au o problemă majoră: sunt predispuși la supra-învățare (overfitting). Dacă sunt lăsați să crească prea mult, ei ajung să memoreze pe de rost exact pacienții din setul de antrenare, creând reguli extrem de specifice care nu se mai aplică la pacienți noi, din lumea reală.
Învățarea de tip Ensemble (Înțelepciunea Mulțimii)
Pentru a rezolva problema supra-învățării, Random Forest aplică Ensemble Learning. Principiul de bază este „înțelepciunea mulțimii”. În loc să ne bazăm diagnosticul pe părerea unui singur medic (un singur arbore de decizie) care poate greși sau poate fi părtinitor, construim un „consiliu medical” format din zeci sau sute de medici (arbori).
La final, pentru a decide soarta unui pacient, fiecare arbore din pădure dă un diagnostic (votează). Modelul final aplică regula majorității: clasa care primește cele mai multe voturi devine predicția finală a modelului.
Secretul diversității: Tehnica Bagging
Dacă toți cei 100 de arbori ar fi antrenați pe exact aceleași date, toți ar învăța aceleași reguli și ar face aceleași greșeli. Consiliul medical ar fi inutil. Pentru ca ansamblul să funcționeze, arborii trebuie să fie diferiți între ei. Aici intervine tehnica numită Bagging (Bootstrap Aggregating):
Bootstrap (Eșantionare aleatoare a pacienților): Fiecare arbore din pădure este antrenat pe un subset diferit de pacienți, extras aleatoriu (cu înlocuire) din setul de date inițial.
Eșantionarea caracteristicilor (Random Subspace): Când un arbore construiește o ramură nouă (pune o întrebare), nu are voie să se uite la toate analizele pacientului (toate coloanele). Algoritmul îi oferă doar o selecție aleatoare de câteva caracteristici (de ex., vede doar vârsta și colesterolul, dar nu vede EKG-ul).
Prin această restricționare intenționată, Random Forest forțează crearea unor arbori foarte diverși. Unii vor deveni „experți” în interpretarea EKG-ului, alții vor detecta tipare ascunse legate de ritmul cardiac și vârstă.
De ce am ales Random Forest pentru acest proiect?
Performanță excelentă pe date complexe: În medicină, relațiile dintre simptome nu sunt mereu liniare. De exemplu, un puls ridicat poate fi normal la o persoană de 20 de ani, dar un semnal de alarmă sever la un pacient de 65 de ani cu istoric de angină. Random Forest detectează natural aceste interacțiuni complexe pe care Regresia Logistică le-ar putea rata.
Robustețe: Fiind un ansamblu bazat pe vot majoritar, este extrem de rezistent la valori aberante (outliers) din setul de date medicale. Dacă un echipament a înregistrat greșit tensiunea arterială a unui pacient, eroarea ar putea păcăli un singur arbore, dar votul majoritar va corecta decizia.
Importanța caracteristicilor (Feature Importance): Deși nu este la fel de transparent ca Regresia Logistică, Random Forest ne poate genera un clasament clar al celor mai importante variabile clinice, arătându-ne care simptome au ajutat cel mai mult la separarea pacienților sănătoși de cei bolnavi.
3.3. Support Vector Machine (SVM): Hiperplan, Marjă Maximă și utilitatea Kernel-urilor
Support Vector Machine (Mașina Vectorilor de Suport) este un algoritm fundamental diferit față de Regresia Logistică sau Random Forest. În timp ce Regresia Logistică gândește în termeni de probabilități, iar Random Forest folosește reguli logice IF-THEN, SVM are o abordare pur geometrică și spațială. El tratează fiecare pacient ca pe un punct (o coordonată) într-un spațiu matematic.
Găsirea graniței perfecte: Hiperplanul și Marja Maximă
Să ne imaginăm o variantă simplificată a problemei noastre, folosind doar două analize medicale: vârsta și colesterolul. Dacă am desena pacienții pe o foaie de hârtie, cei sănătoși ar fi puncte albastre, iar cei bolnavi, puncte roșii. Scopul este să tragem o linie dreaptă care să despartă cele două grupuri.
Această linie poartă denumirea matematică de hiperplan (este o linie în spațiul 2D, o foaie plană în spațiul 3D și un concept abstract în spații cu mai multe dimensiuni).
Problema este că putem trage o infinitate de linii care să separe grupurile. Pe care o alegem? Aici intervine geniul algoritmului SVM: el nu caută orice linie, ci pe aceea care oferă cea mai mare zonă de siguranță (marjă) între cele două clase.
SVM trasează granița astfel încât distanța de la linie până la cel mai apropiat pacient sănătos și cel mai apropiat pacient bolnav să fie maximă.
Această „stradă” de siguranță previne confuziile viitoare. Pacienții (punctele) care stau exact pe marginile acestei străzi sunt cei mai importanți pentru algoritm. Ei se numesc Vectori de Suport (Support Vectors), deoarece ei „susțin” matematic întreaga graniță. Dacă am șterge toate celelalte puncte din baza de date și i-am păstra doar pe ei, granița ar rămâne exact în același loc.
Date nelineare și „Kernel Trick” (Proiecția în 3D)
În realitatea medicală, datele nu sunt aproape niciodată separabile printr-o linie dreaptă. Imaginați-vă că punctele roșii (bolnavii) sunt grupate în centrul foii, fiind complet înconjurate de un inel de puncte albastre (sănătoși). Nicio linie dreaptă desenată pe acea foaie nu poate separa roșu de albastru. Datele sunt nelineare.
Pentru a rezolva problema, SVM folosește un artificiu matematic numit Kernel Trick. În loc să se chinuie să deseneze granițe curbe și complicate pe foaia 2D, algoritmul proiectează datele într-o dimensiune superioară (le duce în 3D).
Vizual, este ca și cum am arunca toate punctele de pe foaie în aer. Prin aplicarea unei funcții matematice (de exemplu, o funcție care ridică mai sus punctele din centru și lasă jos punctele de pe margine), punctele roșii vor ajunge la o altitudine mai mare decât cele albastre.
Odată ce punctele roșii sunt „mai sus” în spațiul 3D, SVM poate pur și simplu să introducă o foaie rigidă de carton (un hiperplan plan) între ele.
Când ne uităm înapoi de sus (în 2D), acea tăietură cu foaia de carton va arăta ca un cerc perfect în jurul bolnavilor.
Cuvântul „Trick” (truc) provine din faptul că algoritmul face toate aceste calcule complexe de separare în 3D (sau chiar în sute de dimensiuni) fără a muta efectiv punctele și fără un consum uriaș de resurse computaționale, folosind funcții speciale precum RBF (Radial Basis Function).
De ce am ales SVM pentru acest proiect?
Performanța pe spații complexe: Setul de date Cleveland are 13 caracteristici clinice diferite. Asta înseamnă că SVM nu caută o linie pe o foaie, ci un hiperplan într-un spațiu cu 13 dimensiuni. Algoritmul excelează în gestionarea acestor spații multidimensionale.
Eficiența pe seturi de date mici: Deoarece decizia finală a modelului se bazează doar pe câțiva pacienți critici (Vectorii de Suport de pe marginea graniței), SVM are o capacitate excelentă de generalizare, chiar dacă setul nostru de date are doar 303 înregistrări.
Flexibilitatea non-liniară: Datorită kernel-ului RBF inclus în optimizarea parametrilor (GridSearchCV), modelul este capabil să descopere granițe medicale subtile (ex: limitele de colesterol sau tensiune care declanșează un risc, în combinație cu vârsta).
Referințe științifice: Trebuie să adaugi minimum 10 referințe la finalul documentului. Exemple de ce să cauți pe Google Scholar: "Machine learning for heart disease prediction review", "Cleveland dataset UCI original paper", "Random forest healthcare classification". Când descrii algoritmii, folosește citări, ex: (Breiman, 2001) pentru Random Forest.
4. Implementarea
Explorarea și Curățarea Datelor (EDA): Am încărcat datele folosind biblioteca pandas. Am identificat valorile lipsă marcate cu caracterul ? și le-am eliminat (dropna). Am binarizat variabila țintă, transformând valorile mai mari ca 0 în 1 (bolnav).
Ingineria Caracteristicilor: Am generat o matrice de corelație pentru a înțelege relațiile liniare dintre variabile. Ulterior, am standardizat caracteristicile continue (precum vârsta și colesterolul) folosind StandardScaler, pas esențial pentru algoritmi bazați pe distanță precum SVM sau Regresia Logistică.
Împărțirea datelor: Am folosit train_test_split pentru a împărți datele în 80% antrenare și 20% testare, folosind un random_state fix pentru reproductibilitate.
Optimizarea parametrilor: Am utilizat GridSearchCV cu 5-fold Cross-Validation pentru a căuta hiperparametrii optimi (ex: parametrul C și kernel pentru SVM, numărul de estimatori și adâncimea maximă pentru Random Forest).
5. Testare și Validare
Metrici folosite: Principala metrică folosită pentru optimizare a fost Acuratețea (Accuracy), deoarece setul de date este relativ echilibrat.
Pentru a evalua performanța algoritmilor antrenați, nu ne-am limitat doar la simpla Acuratețe (Accuracy - procentul total de predicții corecte). Deși acuratețea este o metrică bună pentru a avea o imagine de ansamblu, în domeniul diagnosticului medical ea poate fi înșelătoare, motiv pentru care am analizat și metrici derivate din Matricea de Confuzie (Confusion Matrix): Precizia (Precision), Senzitivitatea (Recall) și scorul F1 (F1-Score).
În contextul prezicerii bolilor de inimă, am acordat o atenție deosebită metricii Recall (Senzitivitate). Aceasta măsoară capacitatea modelului de a identifica corect toți pacienții care sunt într-adevăr bolnavi.
Justificarea medicală a importanței Recall-ului (Minimizarea Fals Negativelor): În orice sistem de triaj medical, erorile de predicție se împart în două categorii, însă impactul lor asupra vieții pacientului este profund asimetric:
Rezultat Fals Pozitiv (Eroare de Tip I): Modelul prezice că pacientul este bolnav, deși el este perfect sănătos. Consecința practică este că pacientul se va speria și va fi trimis să facă investigații suplimentare (un EKG avansat, o ecocardiografie). Costul este unul financiar și de timp, dar viața pacientului nu este pusă în pericol.
Rezultat Fals Negativ (Eroare de Tip II): Modelul prezice că pacientul este perfect sănătos, deși el suferă de o afecțiune cardiacă gravă. Consecința practică este catastrofală: pacientul este trimis acasă fără tratament, riscând un infarct miocardic fatal.
Prin urmare, un model de Machine Learning viabil din punct de vedere clinic trebuie să maximizeze Recall-ul (să reducă rata de Fals Negative spre zero), chiar dacă acest lucru înseamnă o ușoară scădere a Preciziei (acceptarea mai multor Fals Pozitive). Este întotdeauna preferabil să investighezi suplimentar un pacient sănătos decât să ratezi un pacient aflat în pericol. Scorul F1-Score a fost, de asemenea, monitorizat, deoarece oferă o medie armonică între Precizie și Recall, menținând un echilibru corect atunci când ajustăm pragurile de decizie ale algoritmilor.
6. Rezultate și Discuții
6.1. Performanțele Modelelor și Optimizarea
În urma antrenării și evaluării pe setul de testare (reprezentând 20% din datele inițiale, care nu au fost văzute de model în faza de învățare), algoritmii au obținut performanțe notabile. Procesul de optimizare a hiperparametrilor folosind GridSearchCV (Validare Încrucișată în 5 pliuri) a generat următoarele rezultate finale privind acuratețea:
Support Vector Machine (SVM): 90.00% 
Regresia Logistică: 88.33%
Random Forest: 88.33%
Modelul câștigător a fost SVM, atingând o acuratețe de 90.00%. Parametrii optimi identificați pentru acesta au fost C: 0.1 și kernel: 'rbf'. Faptul că nucleul RBF (Radial Basis Function) a depășit Regresia Logistică confirmă ipoteza că relațiile dintre simptomele pacienților și bolile de inimă nu sunt perfect liniare, iar proiectarea datelor într-un spațiu multidimensional a ajutat la găsirea unei granițe de decizie superioare. Valoarea mică a parametrului de regularizare (C = 0.1) indică faptul că modelul a preferat o marjă mai largă, generalizând foarte bine și evitând supra-învățarea (overfitting-ul).
De remarcat este și performanța robustă a Regresiei Logistice (88.33% cu parametrii C: 0.01 și solver: 'lbfgs'), care, în ciuda simplității sale, a reușit să egaleze modelul Random Forest (optimizat la max_depth: 5 și n_estimators: 50).
6.2. Explicabilitatea modelului (Model Explainability - XAI)
Pentru a nu trata modelele ca pe niște „cutii negre” (black-boxes) și pentru a înțelege cum au fost luate deciziile clinice, am aplicat tehnici de explicabilitate asupra modelelor noastre 
Analizând importanța caracteristicilor (Feature Importance din Random Forest) și aplicând valori SHAP (SHapley Additive exPlanations), am putut observa direct contribuția fiecărei analize la predicția finală:
Factori principali de risc: Caracteristicile care au tras cel mai puternic predicțiile în sus (către clasa 1 - Bolnav) au fost anomaliile vizibile pe EKG, defectele reversibile la testul cu taliu (thal), prezența vaselor de sânge colorate la fluoroscopie (ca) și tipul durerii toracice (cp - angina atipică sau tipică).
Factori de protecție / Sănătate: Pe de altă parte, un ritm cardiac maxim ridicat atins în timpul efortului (thalach) și absența anginei induse de efort (exang) au tras constant probabilitatea în jos, indicând o funcție cardiacă sănătoasă. Aceste concluzii algoritmice se aliniază perfect cu literatura medicală din cardiologie, validând faptul că modelul a învățat tipare fiziologice reale, nu doar zgomot statistic.


7. Concluzii și Cunoștințe Noi
Am înțeles importanța impartirii datelor înainte de antrenare și cum optimizarea hiperparametrilor poate crește performanța unui model brut cu 5-10%, alaturi de folosirea unor tehnologii noi precum jupyter notebook.
Limitări: Setul de date este foarte mic (doar 303 pacienți), deci modelul nu poate fi generalizat la populația globală. Datele sunt vechi, din anii '80.
Îmbunătățiri viitoare: Antrenarea pe un set de date modern (ex: dataset-ul CDC) cu mii de pacienți, utilizarea unor rețele neuronale profunde sau implementarea unui sistem de explicabilitate locală cu LIME pentru fiecare pacient în parte.
8. Bibliografie / Referințe Științifice
Originea bazei de date Cleveland: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. The American Journal of Cardiology, 64(5), 304-310.
Algoritmul Random Forest: Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
Algoritmul Support Vector Machine: Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
Regresia Logistică în Medicină: Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (Vol. 398). John Wiley & Sons.
Tehnici Ensemble (Comparare algoritmi): Latha, C. B. C., & Jeeva, S. C. (2019). Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques. Informatics in Medicine Unlocked, 16, 100203.
Machine Learning pentru Boli de Inimă (State of the Art): Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. IEEE Access, 7, 81542-81554.
Optimizarea Hiperparametrilor (GridSearchCV): Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(2), 281-305.
Tehnici de Explicabilitate (SHAP): Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.
Importanța AI-ului Explicabil (XAI) în Sănătate: Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I. (2020). Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. BMC Medical Informatics and Decision Making, 20(1), 1-9.
Impactul general al ML în medicină: Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine, 25(1), 44-56.



    Oldpeak: oldpeak = ST [valoare numerică măsurată prin subdenivelare]

    ST_Slope: panta segmentului ST în vârful efortului [Up: ascendentă, Flat: plată, Down: descendentă]

    HeartDisease: clasa de ieșire (rezultatul) [1: boală de inimă, 0: Normal]
