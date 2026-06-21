# Evroliga – Predviđanje ishoda utakmice – specifikacija projekta

## Opis problema

Cilj ovog projekta je napraviti model koji će predviđati ishod košarkaške utakmice u Evroligi.
Model treba da kaže ko će pobediti – domaći tim ili gostujući tim. Pošto u košarci nerešen
rezultat nije moguć, zadatak je binarna klasifikacija (pobeda domaćina / pobeda gosta).

Predviđanje se vrši na osnovu trenutnih podataka o timovima, samom takmičenju i istoriji
međusobnih susreta.

---

## Skup podataka

Koristiće se javno dostupan skup podataka **Euroleague & Eurocup Datasets** (Kaggle), koji
sadrži podatke o svim utakmicama Evrolige od sezone 2007/2008 do danas, preuzete sa
zvaničnog API-ja Evrolige. Koristi se osnovna tabela utakmica ("Header"), koja sadrži po jedan
red za svaku odigranu utakmicu.

Napomena: ostale tabele iz kolekcije (Boxscore, ShootingGraphic, Points, Comparison,
PlaybyPlay) nisu potrebne za ovaj projekat i mogu se izostaviti.

### Pregled kolona i njihovo značenje

| Kolona | Opis |
|---|---|
| hometeam | tim koji je domaćin utakmice |
| awayteam | tim koji je gost |
| homescore | broj poena koje je postigao domaćin |
| awayscore | broj poena koje je postigao gost |
| result | ciljna promenljiva: ishod utakmice (pobeda domaćina / pobeda gosta) |
| phase | faza takmičenja (regularni deo, plej-of, Final Four) |
| season | sezona Evrolige (npr. 2022/2023) |
| date | datum odigravanja utakmice |
| arena / city | mesto odigravanja utakmice |

---

## Očekivani izlazi (output)

- Evaluacija klasifikacionog modela kroz metrike: tačnost (accuracy), preciznost (precision), odziv (recall) i F1-skor.
- Vizuelna analiza najuticajnijih atributa (feature importance), prikazana grafički putem bar plota.
- Korelaciona matrica atributa – grafički prikaz međusobnih korelacija ulaznih atributa i njihove veze sa ciljnom promenljivom.
- Grafički prikazi distribucije ishoda po fazama takmičenja, odnosa poena domaćina i gosta, kao i uporedni prikaz uspešnosti predviđanja po različitim sezonama.
- Poređenje performansi modela kada se koriste svi atributi nasuprot samo najznačajnijim atributima.
- Tumačenje rezultata sa identifikacijom ključnih karakteristika koje najviše doprinose predviđanju pobede (npr. forma tima, faza takmičenja), uz preporuke za moguća unapređenja modela.

---

## Način evaluacije modela

Model će biti evaluiran hronološkom podelom podataka na trening i test skup (80% starijih
sezona za trening, 20% najnovijih sezona za test), pri čemu će performanse biti merene pomoću
metrika accuracy, precision, recall i F1-skor. Dodatno, rezultati će biti analizirani pomoću
konfuzione matrice.

Hiperparametri svakog modela biće podešeni korišćenjem **GridSearchCV** u kombinaciji sa
**TimeSeriesSplit** kros-validacijom (5 foldova), koja poštuje vremenski redosled podataka i
sprečava curenje informacija iz budućih sezona u validacioni skup.

Model će biti implementiran korišćenjem standardnih algoritama mašinskog učenja za
klasifikaciju: Logistic Regression, Decision Tree, XGBoost i Random Forest.

Konačan model bira se na osnovu **F1 macro** skora, koji jednako vrednuje obe klase
(pobeda domaćina i pobeda gosta), čime se izbegava pristrasnost prema dominantnoj klasi.

---

## Zahtevi za procesiranje podataka

- **Nedostajuće vrednosti** – ukoliko postoje u kolonama poput arena ili attendance, biće
  obrađene na način koji ne narušava performanse modela (popunjavanje medijanom za numeričke
  atribute, najčešćom vrednošću za kategorijske).

- **Kategorijski atributi** koji budu korišćeni u modelu (hometeam, awayteam, phase) biće
  konvertovani u numerički format metodom Label Encoding. Pošto isti klub kroz sezone nastupa
  pod različitim sponzorskim imenima, primenjen je algoritam kanonizacije naziva timova koji
  grupiše sve istorijske varijante jednog kluba pod jedinstveni naziv.

- **Uklanjanje nebitnih kolona** – atributi koji nisu relevantni za predikciju (date, arena, city)
  biće uklonjeni iz skupa podataka.

- **Kolone homescore i awayscore** neće se koristiti kao ulazni atributi za predviđanje jer su
  ti podaci vidljivi tek nakon završetka same utakmice. Na osnovu njih se izvodi ciljna
  promenljiva (result), kao i atributi forme.

- **Kreiranje novih atributa** – u cilju poboljšanja performansi modela, biće dodati sledeći
  atributi koji se izvode iz postojećih:
  - `home_form` – prosečan broj poena domaćina u svim prethodnim utakmicama **tekuće sezone**
    (kumulativni prosek koji raste sa svakim kolom; forma se resetuje na početku svake sezone)
  - `away_form` – prosečan broj poena gosta u svim prethodnim utakmicama tekuće sezone
  - `head_to_head_advantage` – broj pobeda domaćina u svim prethodnim međusobnim susretima,
    uzimajući u obzir sve istorijske nazive oba kluba

- **year** – godina održavanja sezone, izdvojena iz atributa season i predstavljena u numeričkom formatu.

---

## Ulazni atributi

| Atribut | Opis |
|---|---|
| hometeam | domaći tim (label encoded) |
| awayteam | gostujući tim (label encoded) |
| phase | faza takmičenja (label encoded) |
| year | godina sezone |
| home_form | kumulativni prosečan broj poena domaćina u tekućoj sezoni |
| away_form | kumulativni prosečan broj poena gosta u tekućoj sezoni |
| head_to_head_advantage | broj pobeda domaćina u prethodnim međusobnim susretima |

## Izlaz

- `result` – pobeda domaćina (1) / pobeda gosta (0)

---

## Deployment modela

Finalni model se eksportuje u `.pkl` format i dostupan je kroz:
- **Web aplikaciju** (Streamlit) – interaktivni UI za odabir timova, sezone i faze takmičenja
  sa prikazom vjerovatnoća ishoda u realnom vremenu
- **CLI sučelje** – predikcija iz komandne linije za brzo testiranje

---

## Dokumentovanje rezultata

Rezultati će biti prikazani u Word dokumentu kroz tabele sa metrikama (accuracy, precision,
recall i F1-skor), grafike za analizu podataka, prikaz najvažnijih atributa i konfuzionu matricu.
Konačan izbor modela biće donet na osnovu poređenja F1 macro skora svih testiranih algoritama.
