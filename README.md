# Euroleague – Predviđanje ishoda utakmice

Projekat za predmet SAUSAU 2026.  
Binarni klasifikacioni model koji predviđa pobednika košarkaške utakmice u Evroligi (domaći / gost).

---

## Struktura projekta

```
ml-project/
├── data/
│   ├── raw/          ← ovde stavite preuzeti CSV sa Kaggle-a
│   └── processed/    ← automatski generisano
├── src/
│   ├── data_preparation.py   # učitavanje, čišćenje, feature engineering
│   ├── train.py              # treniranje 4 modela + GridSearchCV
│   ├── evaluate.py           # metrike, grafici, feature importance
│   └── predict.py            # predikcija jedne utakmice (CLI + modul)
├── models/                   ← automatski generisano (sačuvani modeli)
├── results/
│   ├── figures/              ← svi grafici (.png)
│   └── metrics/              ← CSV i JSON sa metrikama
├── app/
│   └── app.py                # Streamlit web aplikacija
├── main.py                   # orchestrator – pokreće ceo pipeline
└── requirements.txt
```

---

## Postavljanje

```bash
pip install -r requirements.txt
```

---

## Podaci

1. Preuzmite skup podataka sa Kaggle-a:  
   **Euroleague & Eurocup Datasets** – `babissamothrakis/euroleague-datasets`
2. Smestite fajl `EL_games_header.csv` (ili sličan Header CSV) u folder `data/raw/`.

---

## Pokretanje

### Ceo pipeline odjednom
```bash
python main.py all
```
Ili sa eksplicitnim putanjom do CSV-a:
```bash
python main.py all --data data/raw/EL_games_header.csv
```

### Korak po korak
```bash
python main.py prepare    # preprocesiranje i feature engineering
python main.py train      # treniranje i podešavanje hiperparametara
python main.py evaluate   # metrike i grafici
```

### Web aplikacija
```bash
streamlit run app/app.py
```

### Predikcija iz komandne linije
```bash
python src/predict.py --home "Fenerbahce" --away "Real Madrid" --phase RS --year 2024

# Prikazati dostupne timove i faze:
python src/predict.py --list-teams
```

---

## Ulazni atributi modela

| Atribut | Opis |
|---|---|
| `hometeam` | domaći tim (label encoded) |
| `awayteam` | gostujući tim (label encoded) |
| `phase` | faza takmičenja (RS / PO / F4) |
| `year` | godina sezone |
| `home_form` | prosečan broj poena domaćina u poslednje 3 utakmice |
| `away_form` | prosečan broj poena gosta u poslednje 3 utakmice |
| `head_to_head_advantage` | broj pobeda domaćina u prethodnim međusobnim susretima |

> **Napomena:** `homescore` i `awayscore` se **ne koriste** kao ulazni atributi  
> (dostupni su tek nakon odigrane utakmice).

---

## Algoritmi

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Svi modeli se podešavaju pomoću `GridSearchCV` sa 5-fold stratified cross-validation.

---

## Izlazi

- `results/metrics/all_metrics.csv` – tačnost, preciznost, odziv, F1 po modelu
- `results/metrics/classification_reports.txt` – detaljni izveštaji
- `results/figures/confusion_matrices.png`
- `results/figures/metrics_comparison.png`
- `results/figures/feature_importance.png`
- `results/figures/feature_selection_rf.png`
- `results/figures/correlation_matrix.png`
- `results/figures/outcome_distribution.png`
- `results/figures/score_distribution.png`
- `results/figures/accuracy_by_season.png`
