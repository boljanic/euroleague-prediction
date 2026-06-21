"""
Prediction interface: load the best saved model and predict a single game.
Can be used as a module (imported by app.py) or run from the CLI.
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR         = os.path.join(BASE_DIR, 'models')

# Manual overrides: VARIANT → CANONICAL_DISPLAY_NAME
# Covers cases the algorithm can't detect (different sponsor names, spelling variants,
# historical renames, trailing spaces, MILAN vs MILANO, etc.)
_MANUAL_CANONICAL = {
    # FENERBAHCE (fix: FB DOGUS wrongly merged Fenerbahce + Darussafaka)
    'FB DOGUS':                            'FENERBAHCE BEKO ISTANBUL',
    'FENERBAHCE DOGUS ISTANBUL':           'FENERBAHCE BEKO ISTANBUL',
    'FENERBAHCE ISTANBUL':                 'FENERBAHCE BEKO ISTANBUL',
    'FENERBAHCE ULKER':                    'FENERBAHCE BEKO ISTANBUL',
    'FENERBAHCE ULKER ISTANBUL':           'FENERBAHCE BEKO ISTANBUL',
    'FENERBAHCE BEKO ISTANBUL':            'FENERBAHCE BEKO ISTANBUL',
    # DARUSSAFAKA
    'DARUSSAFAKA DOGUS ISTANBUL':          'DARUSSAFAKA TEKFEN ISTANBUL',
    'DARUSSAFAKA TEKFEN ISTANBUL':         'DARUSSAFAKA TEKFEN ISTANBUL',
    # FC BARCELONA (all sponsor/name variants + trailing spaces)
    'AXA FC BARCELONA':                    'FC BARCELONA',
    'FC BARCELONA LASSA':                  'FC BARCELONA',
    'FC BARCELONA ':                       'FC BARCELONA',
    'FC BARCELONA REGAL':                  'FC BARCELONA',
    'REGAL FC BARCELONA':                  'FC BARCELONA',
    'REGAL FC BARCELONA ':                 'FC BARCELONA',
    'FC BARCELONA':                        'FC BARCELONA',
    # BC KHIMKI
    'BC KHIMKI':                           'KHIMKI MOSCOW REGION',
    'BC KHIMKI MOSCOW REGION':             'KHIMKI MOSCOW REGION',
    'KHIMKI MOSCOW REGION':                'KHIMKI MOSCOW REGION',
    # EA7 OLIMPIA MILANO (MILAN == MILANO, same club)
    'AX ARMANI EXCHANGE MILAN':            'EA7 EMPORIO ARMANI MILANO',
    'EA7 EMPORIO ARMANI MILAN':            'EA7 EMPORIO ARMANI MILANO',
    'AX ARMANI EXCHANGE OLIMPIA MILAN':    'EA7 EMPORIO ARMANI MILANO',
    'ARMANI JEANS MILANO':                 'EA7 EMPORIO ARMANI MILANO',
    'MILANO':                              'EA7 EMPORIO ARMANI MILANO',
    'EA7 EMPORIO ARMANI MILANO':           'EA7 EMPORIO ARMANI MILANO',
    # KK ZAGREB / CEDEVITA ZAGREB
    'KK ZAGREB':                           'CEDEVITA ZAGREB',
    'KK ZAGREB CROATIA OSIGURANJE':        'CEDEVITA ZAGREB',
    'CEDEVITA ZAGREB':                     'CEDEVITA ZAGREB',
    # BASKONIA — all eras (Tau Ceramica, Caja Laboral, Laboral Kutxa, Cazoo, ...)
    'BASKONIA':                            'BASKONIA VITORIA-GASTEIZ',
    'BASKONIA VITORIA GASTEIZ':            'BASKONIA VITORIA-GASTEIZ',
    'BASKONIA VITORIA-GASTEIZ':            'BASKONIA VITORIA-GASTEIZ',
    'BITCI BASKONIA VITORIA-GASTEIZ':      'BASKONIA VITORIA-GASTEIZ',
    'CAZOO':                               'BASKONIA VITORIA-GASTEIZ',
    'CAZOO BASKONIA VITORIA-GASTEIZ':      'BASKONIA VITORIA-GASTEIZ',
    'CAJA LABORAL':                        'BASKONIA VITORIA-GASTEIZ',
    'CAJA LABORAL BASKONIA':               'BASKONIA VITORIA-GASTEIZ',
    'CAJA LABORAL VITORIA':                'BASKONIA VITORIA-GASTEIZ',
    'KIROLBET BASKONIA VITORIA GASTEIZ':   'BASKONIA VITORIA-GASTEIZ',
    'KIROLBET BASKONIA VITORIA-GASTEIZ':   'BASKONIA VITORIA-GASTEIZ',
    'KOSNER BASKONIA VITORIA-GASTEIZ':     'BASKONIA VITORIA-GASTEIZ',
    'LABORAL KUTXA VITORIA':               'BASKONIA VITORIA-GASTEIZ',
    'LABORAL KUTXA VITORIA GASTEIZ':       'BASKONIA VITORIA-GASTEIZ',
    'TAU CERAMICA':                        'BASKONIA VITORIA-GASTEIZ',
    'TD SYSTEMS BASKONIA VITORIA-GASTEIZ': 'BASKONIA VITORIA-GASTEIZ',
    # ASVEL (Lyon era + LDLC era = same club)
    'ASVEL LYON':                          'LDLC ASVEL VILLEURBANNE',
    'LDLC':                                'LDLC ASVEL VILLEURBANNE',
    'LDLC ASVEL VILLEURBANNE':             'LDLC ASVEL VILLEURBANNE',
    # ROANNE
    'ROANNE':                              'CHORALE ROANNE',
    # UNICAJA
    'BALONCESTO MALAGA':                   'UNICAJA MALAGA',
    'UNICAJA':                             'UNICAJA MALAGA',
    # DKV JOVENTUT (trailing space in raw data)
    'DKV JOVENTUT ':                       'DKV JOVENTUT',
    # BILBAO BASKET
    'GESCRAP BB':                          'BILBAO BASKET',
    'BIZKAIA BILBAO BASKET':               'BILBAO BASKET',
    # BROSE BAMBERG (Baskets Bamberg = isti klub, stariji naziv koji se pojavljuje samo kao domaćin)
    'BASKETS BAMBERG':                     'BROSE BASKETS BAMBERG',
    'BROSE BAMBERG':                       'BROSE BASKETS BAMBERG',
    'BROSE BASKETS':                       'BROSE BASKETS BAMBERG',
    'BROSE BASKETS BAMBERG':               'BROSE BASKETS BAMBERG',
}

# Words that identify a city/location (not the club name)
_CITY_WORDS = {
    'ISTANBUL', 'MADRID', 'ATHENS', 'MILAN', 'MILANO', 'BELGRADE', 'BERLIN',
    'MUNICH', 'BARCELONA', 'VITORIA', 'GASTEIZ', 'MOSCOW', 'TEL', 'AVIV',
    'KAUNAS', 'VILNIUS', 'ZAGREB', 'KAZAN', 'ROMA', 'ROME', 'PIRAEUS',
    'THESSALONIKI', 'GDYNIA', 'SOPOT', 'VILLEURBANNE', 'LYON', 'OLDENBURG',
    'BAMBERG', 'ZGORZELEC', 'ZIELONA', 'GORA', 'PETERSBURG', 'KRASNODAR',
    'SASSARI', 'NANTERRE', 'NANCY', 'ROANNE', 'LIMOGES', 'STRASBOURG',
    'CHARLEROI', 'PODGORICA', 'KIEV', 'IZMIR', 'AVELLINO', 'MONACO',
    'PARIS', 'DUBAI', 'KLAIPEDA', 'NIZHNY', 'NOVGOROD', 'MAROUSSI',
    'CANARIA', 'VALENCIA', 'MALAGA', 'SIENA', 'BOLOGNA', 'SARAJEVO',
    'KHIMKI', 'REGION', 'ST',
}

# Generic words that don't identify a club
_GENERIC_WORDS = {'BC', 'BK', 'KK', 'BASKET', 'BASKETBALL', 'BALONCESTO', 'CLUB'}


def _team_keywords(name):
    words = set(name.replace('-', ' ').replace(':', ' ').split())
    return frozenset(w for w in words
                     if w not in _CITY_WORDS and w not in _GENERIC_WORDS and len(w) > 2)


def _team_city(name):
    words = set(name.replace('-', ' ').replace(':', ' ').split())
    return frozenset(w for w in words if w in _CITY_WORDS)


def build_team_clusters(team_list, processed_df=None):
    """
    Group team names that represent the same club (different sponsors over seasons).

    Applies _MANUAL_CANONICAL first (deterministic, covers known edge-cases), then
    falls back to keyword-similarity clustering for any unmapped names.

    Returns:
        display_to_variants  {display_name: [all_variant_names]}
        variant_to_display   {any_variant:  display_name}

    The display name is the most recently used variant in processed_df.
    """
    # --- Step 1: manual overrides ---
    manual_groups = {}   # canonical → [variants]
    remaining     = []
    team_set      = set(team_list)

    for team in team_list:
        canonical = _MANUAL_CANONICAL.get(team)
        if canonical is not None:
            manual_groups.setdefault(canonical, [])
            if team not in manual_groups[canonical]:
                manual_groups[canonical].append(team)
        else:
            remaining.append(team)

    # Ensure canonical target itself is included if it exists in team_list
    for canonical, variants in manual_groups.items():
        if canonical in team_set and canonical not in variants:
            variants.append(canonical)

    # --- Step 2: algorithmic clustering for unmapped teams ---
    teams_sorted = sorted(remaining, key=lambda t: (len(_team_keywords(t)), len(t)))
    algo_groups  = {}   # seed → [variants]

    for team in teams_sorted:
        kw   = _team_keywords(team)
        city = _team_city(team)

        best_seed, best_score = None, 0.0
        for seed in algo_groups:
            skw   = _team_keywords(seed)
            scity = _team_city(seed)
            if city and scity and city != scity:
                continue
            if not kw or not skw:
                continue
            inter = kw & skw
            if not inter:
                continue
            score = len(inter) / min(len(kw), len(skw))
            if score >= 0.5 and score > best_score:
                best_seed, best_score = seed, score

        if best_seed:
            algo_groups[best_seed].append(team)
        else:
            algo_groups[team] = [team]

    # --- Step 3 & 4: build final display_to_variants ---
    display_to_variants = {}
    variant_to_display  = {}
    manual_variant_set  = {v for vs in manual_groups.values() for v in vs}

    # Manual groups: canonical target IS the display name (deterministic, no year heuristic)
    for canonical, variants in manual_groups.items():
        seen   = set()
        unique = [v for v in variants if not (v in seen or seen.add(v))]
        display_to_variants[canonical] = unique
        for v in unique:
            variant_to_display[v] = canonical

    # Algo groups: use most recently used variant as display name,
    # but skip any team already assigned by a manual group
    for seed, variants in algo_groups.items():
        if seed in manual_variant_set:
            continue
        new_variants = [v for v in variants if v not in manual_variant_set]
        if not new_variants:
            continue

        seen   = set()
        unique = [v for v in new_variants if not (v in seen or seen.add(v))]

        display = seed
        if processed_df is not None and 'year' in processed_df.columns:
            best_year = -1
            for v in unique:
                mask = (processed_df['hometeam'] == v) | (processed_df['awayteam'] == v)
                if mask.any():
                    yr = int(processed_df.loc[mask, 'year'].max())
                    if yr > best_year:
                        best_year, display = yr, v

        if display not in display_to_variants:
            display_to_variants[display] = unique
            for v in unique:
                variant_to_display[v] = display

    return display_to_variants, variant_to_display


def _resolve_for_encoding(display_name, clusters, year):
    """
    Map a canonical display name back to the exact variant that the label encoder
    knows about, preferring the variant used in the requested season year.
    """
    variants = clusters.get(display_name, [display_name])
    if len(variants) == 1:
        return variants[0]

    try:
        df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv'))
        if year is not None and 'year' in df.columns:
            yr_names = set(df[df['year'] == year][['hometeam', 'awayteam']].values.ravel())
            for v in variants:
                if v in yr_names:
                    return v
        # Fallback: most recently used variant overall
        if 'year' in df.columns:
            for yr in sorted(df['year'].dropna().unique(), reverse=True):
                yr_names = set(df[df['year'] == yr][['hometeam', 'awayteam']].values.ravel())
                for v in variants:
                    if v in yr_names:
                        return v
    except Exception:
        pass

    return display_name


def load_best_model():
    """Return (model, encoders, scaler, metadata, model_name)."""
    with open(os.path.join(MODELS_DIR, 'best_model.json')) as f:
        best_name = json.load(f)['best_model']

    with open(os.path.join(MODELS_DIR, f'{best_name}.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(DATA_PROCESSED_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    return model, encoders, scaler, metadata, best_name


def get_available_values():
    """Return teams (deduplicated), phases, years, and the cluster mapping."""
    with open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv'))

    _empty = type('E', (), {'classes_': []})()
    all_teams = sorted(set(
        encoders.get('hometeam', _empty).classes_.tolist() +
        encoders.get('awayteam', _empty).classes_.tolist()
    ))
    years = sorted(df['year'].dropna().unique().astype(int).tolist()) if 'year' in df.columns else []

    clusters, variant_to_display = build_team_clusters(all_teams, processed_df=df)

    return {
        'teams':            sorted(clusters.keys()),
        'team_clusters':    clusters,
        'variant_to_display': variant_to_display,
        'phases':           sorted(encoders['phase'].classes_.tolist()) if 'phase' in encoders else [],
        'years':            years,
    }


def get_team_stats(hometeam, awayteam, year=None, clusters=None):
    """
    Look up home_form, away_form, and head_to_head_advantage from the processed
    dataset using only data up to and including the selected season (no future leakage).
    Searches across all historical club variants when clusters is provided.
    """
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv'))
    if 'date' in df.columns:
        df = df.sort_values('date')

    home_variants = clusters.get(hometeam, [hometeam]) if clusters else [hometeam]
    away_variants = clusters.get(awayteam, [awayteam]) if clusters else [awayteam]

    # Only use data up to (and including) the selected year — no future seasons
    df_past = df[df['year'] <= year] if (year is not None and 'year' in df.columns) else df
    if df_past.empty:
        df_past = df

    home_rows = df_past[df_past['hometeam'].isin(home_variants)]
    home_form = (float(home_rows['home_form'].iloc[-1])
                 if not home_rows.empty else float(df['home_form'].mean()))

    away_rows = df_past[df_past['awayteam'].isin(away_variants)]
    away_form = (float(away_rows['away_form'].iloc[-1])
                 if not away_rows.empty else float(df['away_form'].mean()))

    # H2H: recompute from results (not the precomputed column) so name-variant
    # changes across seasons don't break the count.
    # Counts all wins by the home team (any variant) over the away team (any variant),
    # regardless of which team was home/away in each historical game.
    h2h_all = df_past[
        (df_past['hometeam'].isin(home_variants) & df_past['awayteam'].isin(away_variants)) |
        (df_past['hometeam'].isin(away_variants) & df_past['awayteam'].isin(home_variants))
    ]
    h2h = int(
        ((h2h_all['hometeam'].isin(home_variants)) & (h2h_all['result'] == 1)).sum() +
        ((h2h_all['awayteam'].isin(home_variants)) & (h2h_all['result'] == 0)).sum()
    )

    return {'home_form': home_form, 'away_form': away_form, 'h2h': h2h}


def _safe_encode(encoders, col, val, fallbacks=None):
    """Encode a value; try cluster fallbacks before using median class."""
    if col not in encoders:
        return 0
    le = encoders[col]
    if val in le.classes_:
        return int(le.transform([val])[0])
    if fallbacks:
        for fb in fallbacks:
            if fb in le.classes_:
                return int(le.transform([fb])[0])
    print(f"Warning: '{val}' not in training data for '{col}'. Using fallback.")
    return len(le.classes_) // 2


def _default_form():
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv'))
    return float((df['home_form'].mean() + df['away_form'].mean()) / 2)


def predict_game(hometeam, awayteam, phase, year,
                 home_form=None, away_form=None, h2h=0, clusters=None):
    """
    Predict the outcome of a single game.

    hometeam / awayteam may be canonical display names; pass clusters so the
    function can resolve them to the exact encoder-known variant.
    """
    model, encoders, scaler, metadata, model_name = load_best_model()
    feature_cols = metadata['feature_cols']

    default_form = _default_form()
    if home_form is None:
        home_form = default_form
    if away_form is None:
        away_form = default_form

    home_enc = _resolve_for_encoding(hometeam, clusters, year) if clusters else hometeam
    away_enc = _resolve_for_encoding(awayteam, clusters, year) if clusters else awayteam

    home_variants = clusters.get(hometeam, [hometeam]) if clusters else [hometeam]
    away_variants = clusters.get(awayteam, [awayteam]) if clusters else [awayteam]

    feature_map = {
        'hometeam_encoded':       _safe_encode(encoders, 'hometeam', home_enc, fallbacks=home_variants),
        'awayteam_encoded':       _safe_encode(encoders, 'awayteam', away_enc, fallbacks=away_variants),
        'phase_encoded':          _safe_encode(encoders, 'phase', phase),
        'year':                   int(year),
        'home_form':              float(home_form),
        'away_form':              float(away_form),
        'head_to_head_advantage': int(h2h),
    }

    X = pd.DataFrame([[feature_map.get(f, 0) for f in feature_cols]], columns=feature_cols)

    if model_name == 'logistic_regression':
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    home_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

    return {
        'prediction':           'Home Win' if pred == 1 else 'Away Win',
        'home_win_probability': home_prob,
        'away_win_probability': 1.0 - home_prob,
        'model_used':           model_name,
    }


def main():
    parser = argparse.ArgumentParser(description='Predict a Euroleague game outcome.')
    parser.add_argument('--home',       required=True,  help='Home team name')
    parser.add_argument('--away',       required=True,  help='Away team name')
    parser.add_argument('--phase',      default='REGULAR SEASON')
    parser.add_argument('--year',       type=int, default=2024)
    parser.add_argument('--home-form',  type=float, default=None)
    parser.add_argument('--away-form',  type=float, default=None)
    parser.add_argument('--h2h',        type=int,   default=0)
    parser.add_argument('--list-teams', action='store_true')
    args = parser.parse_args()

    if args.list_teams:
        av = get_available_values()
        print("Teams (deduplicated):"); [print(f"  {t}") for t in av['teams']]
        print("Phases:"); [print(f"  {p}") for p in av['phases']]
        return

    av = get_available_values()
    result = predict_game(
        hometeam=args.home, awayteam=args.away,
        phase=args.phase, year=args.year,
        home_form=args.home_form, away_form=args.away_form,
        h2h=args.h2h, clusters=av['team_clusters'],
    )
    print(f"\n{'='*45}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Home win   : {result['home_win_probability']:.1%}")
    print(f"  Away win   : {result['away_win_probability']:.1%}")
    print(f"  Model      : {result['model_used']}")
    print('='*45)


if __name__ == '__main__':
    main()
