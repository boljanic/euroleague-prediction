import sys
sys.path.insert(0, 'ml-project')
from src.predict import get_available_values
import pandas as pd

av = get_available_values()
clusters = av['team_clusters']

print(f"Timova u dropdownu: {len(av['teams'])}")
print()

checks = [
    'FC BARCELONA', 'KHIMKI MOSCOW REGION', 'EA7 EMPORIO ARMANI MILANO',
    'CEDEVITA ZAGREB', 'BASKONIA VITORIA-GASTEIZ', 'FENERBAHCE BEKO ISTANBUL',
    'DARUSSAFAKA TEKFEN ISTANBUL', 'LDLC ASVEL VILLEURBANNE', 'CHORALE ROANNE',
    'UNICAJA MALAGA', 'BILBAO BASKET', 'DKV JOVENTUT',
]
for name in checks:
    variants = clusters.get(name, [])
    print(f"{name} ({len(variants)} varijante):")
    for v in variants:
        print(f"   - {v}")
    print()

df = pd.read_csv('ml-project/data/processed/processed_data.csv')
print(f"Ukupno utakmica u bazi: {len(df)}")

print()
print("Preostali klasteri sa vise varijanti:")
for d, vs in sorted(clusters.items()):
    if len(vs) > 1:
        print(f"  {d}: {len(vs)} varijante")
