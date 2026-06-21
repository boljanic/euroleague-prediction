"""
Data preparation pipeline: loading, cleaning, feature engineering, encoding.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Maps possible column name variants to standardized names
COLUMN_ALIASES = {
    'hometeam':  ['hometeam', 'home_team', 'localclub', 'local_club', 'home', 'team_a'],
    'awayteam':  ['awayteam', 'away_team', 'roadclub', 'road_club', 'away', 'team_b'],
    'homescore': ['homescore', 'home_score', 'localclubscore', 'localScore', 'hscore', 'score_a'],
    'awayscore': ['awayscore', 'away_score', 'roadclubscore', 'roadScore', 'ascore', 'score_b'],
    'phase':     ['phase', 'Phase', 'competition_phase', 'round_phase'],
    'season':    ['season', 'Season', 'competition_season', 'season_code'],
    'date':      ['date', 'Date', 'gametime', 'Gametime', 'game_date', 'GameTime'],
    'arena':     ['arena', 'Arena', 'venue', 'stadium'],
    'city':      ['city', 'City', 'location'],
}


def normalize_columns(df):
    """Rename dataset columns to the standard names used throughout this project."""
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_map and standard not in df.columns:
                rename[lower_map[alias.lower()]] = standard
                break
    if rename:
        print(f"Renaming columns: {rename}")
        df = df.rename(columns=rename)
    return df


def load_data(filepath=None):
    """Locate and load the raw CSV file."""
    if filepath is None:
        candidates = [f for f in os.listdir(DATA_RAW_DIR) if f.lower().endswith('.csv')]
        # Prefer files with 'header' in the name (spec uses the Header table)
        header_files = [f for f in candidates if 'header' in f.lower()]
        chosen = header_files[0] if header_files else (candidates[0] if candidates else None)
        if chosen is None:
            raise FileNotFoundError(
                f"No CSV found in {DATA_RAW_DIR}. "
                "Download the Kaggle dataset and place the Header CSV there."
            )
        filepath = os.path.join(DATA_RAW_DIR, chosen)

    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    return df


def explore_data(df):
    """Print a short EDA summary to the console."""
    print(f"\n{'='*50}")
    print("EXPLORATORY DATA ANALYSIS")
    print('='*50)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    if 'result' in df.columns:
        print(f"\nResult distribution:\n{df['result'].value_counts()}")
    if 'phase' in df.columns:
        print(f"\nPhase distribution:\n{df['phase'].value_counts()}")
    return df


def normalize_phases(df):
    """Merge duplicate/rare phase labels into canonical names."""
    if 'phase' not in df.columns:
        return df
    phase_map = {
        'PLAY OFF':    'PLAYOFFS',
        'TOP SIXTEEN': 'TOP 16',
        'PLAY-IN':     'PLAYOFFS',  # only 9 games, model can't learn it separately
    }
    before = df['phase'].value_counts().to_dict()
    df['phase'] = df['phase'].replace(phase_map)
    after = df['phase'].value_counts().to_dict()
    print(f"Phase normalization: {before} -> {after}")
    return df


def handle_missing_values(df):
    """Fill missing values without dropping rows where possible."""
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        if df[col].dtype in ['float64', 'int64']:
            fill = df[col].median()
        else:
            mode = df[col].mode()
            fill = mode[0] if len(mode) > 0 else 'Unknown'
        df[col] = df[col].fillna(fill)
        print(f"  Filled {n_missing} missing values in '{col}' with {fill!r}")
    return df


def prepare_result_column(df):
    """
    Normalise the target column to integers: 1 = home win, 0 = away win.
    If the column does not exist it is derived from scores.
    """
    if 'result' not in df.columns:
        if 'homescore' in df.columns and 'awayscore' in df.columns:
            df['result'] = (df['homescore'] > df['awayscore']).astype(int)
            print("Created 'result' column from homescore vs awayscore.")
            return df
        raise ValueError("Cannot determine result: missing 'result', 'homescore', or 'awayscore'.")

    sample = set(str(v).strip().lower() for v in df['result'].dropna().unique())
    if sample <= {'home', 'away'}:
        df['result'] = df['result'].str.strip().str.lower().map({'home': 1, 'away': 0})
    elif sample <= {'h', 'a'}:
        df['result'] = df['result'].str.strip().str.upper().map({'H': 1, 'A': 0})
    elif sample <= {'local', 'road'}:
        df['result'] = df['result'].str.strip().str.lower().map({'local': 1, 'road': 0})
    elif sample <= {'1', '0', '1.0', '0.0'}:
        df['result'] = df['result'].astype(float).astype(int)
    else:
        # Fallback: re-derive from scores
        if 'homescore' in df.columns and 'awayscore' in df.columns:
            df['result'] = (df['homescore'] > df['awayscore']).astype(int)
            print("Re-derived 'result' from scores (unknown encoding detected).")
        else:
            raise ValueError(f"Unknown result encoding: {sample}")

    print(f"Result distribution:\n{df['result'].value_counts()}")
    return df


def extract_year(df):
    """Extract the start year of the season as a numeric feature."""
    if 'season' in df.columns:
        # Handles '2022-23', '2022/2023', '2022-2023', 'E2022', etc.
        df['year'] = (df['season'].astype(str)
                                  .str.extract(r'(\d{4})')[0]
                                  .astype(float)
                                  .astype('Int64'))
    elif 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    else:
        df['year'] = 2020  # fallback constant

    median_year = int(df['year'].dropna().median())
    df['year'] = df['year'].fillna(median_year).astype(int)
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    return df


def compute_team_forms(df, n_games=3):
    """
    Rolling average of points in last n_games for the home and away team.
    Uses shift(1) to avoid including the current game (no data leakage).
    """
    df = df.sort_values('date').reset_index(drop=True)
    global_avg = (df['homescore'].mean() + df['awayscore'].mean()) / 2

    yr_cols = ['year'] if 'year' in df.columns else []
    home_rows = (df[['date', 'hometeam', 'homescore'] + yr_cols]
                 .rename(columns={'hometeam': 'team', 'homescore': 'score'}))
    away_rows = (df[['date', 'awayteam', 'awayscore'] + yr_cols]
                 .rename(columns={'awayteam': 'team', 'awayscore': 'score'}))

    all_games = (pd.concat([home_rows, away_rows], ignore_index=True)
                 .sort_values(['team', 'date'])
                 .reset_index(drop=True))

    # Group by (team, year) so form resets at the start of each season.
    # A team's 2017 form is irrelevant to their 2025 campaign.
    group_cols = ['team', 'year'] if 'year' in all_games.columns else ['team']
    all_games['form'] = (all_games
                         .groupby(group_cols)['score']
                         .transform(lambda x: x.shift(1).expanding(min_periods=1).mean()))
    all_games['form'] = all_games['form'].fillna(global_avg)

    # One form value per (team, date) — use the value before any same-day games
    form_lookup = (all_games
                   .groupby(['team', 'date'])['form']
                   .first()
                   .reset_index())

    df = df.merge(form_lookup.rename(columns={'team': 'hometeam', 'form': 'home_form'}),
                  on=['hometeam', 'date'], how='left')
    df = df.merge(form_lookup.rename(columns={'team': 'awayteam', 'form': 'away_form'}),
                  on=['awayteam', 'date'], how='left')

    df['home_form'] = df['home_form'].fillna(global_avg)
    df['away_form'] = df['away_form'].fillna(global_avg)

    print(f"home_form stats: mean={df['home_form'].mean():.1f}, "
          f"min={df['home_form'].min():.1f}, max={df['home_form'].max():.1f}")
    return df


def compute_head_to_head(df):
    """
    Number of times the current home team has beaten the current away team
    in ALL previous meetings (regardless of home/away role).
    O(n) accumulation — no row-by-row filtering.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # win_record[(winner, loser)] = count
    win_record = {}
    h2h_values = []

    for idx in range(len(df)):
        home = df.at[idx, 'hometeam']
        away = df.at[idx, 'awayteam']
        result = df.at[idx, 'result']

        # Wins of home team vs away team (as either home or away in previous games)
        home_wins = win_record.get((home, away), 0)
        h2h_values.append(home_wins)

        # Update running tally
        if result == 1:  # home won
            win_record[(home, away)] = win_record.get((home, away), 0) + 1
        else:            # away won
            win_record[(away, home)] = win_record.get((away, home), 0) + 1

    df['head_to_head_advantage'] = h2h_values
    print(f"head_to_head_advantage max: {df['head_to_head_advantage'].max()}")
    return df


def aggregate_box_score_to_games(df):
    """
    Convert a player-level box-score DataFrame to one game-level row per game.

    Detects home/away roles from the 'game' column (format 'HOME-AWAY').
    Sums player points to produce homescore/awayscore, then drops the
    player-level columns so the rest of the pipeline sees a Header-style table.
    """
    print("Box-score format detected - aggregating player rows to game level...")

    group_cols = [c for c in ['game_id', 'game', 'team_id', 'phase', 'season', 'round']
                  if c in df.columns]
    agg_df = df.groupby(group_cols, as_index=False).agg(score=('points', 'sum'))

    if 'game' in agg_df.columns:
        split = agg_df['game'].str.split('-', n=1, expand=True)
        agg_df['_home_abbr'] = split[0].str.strip().str.upper()
    else:
        agg_df['_home_abbr'] = ''

    records = []
    for game_id, grp in agg_df.groupby('game_id'):
        teams = grp.reset_index(drop=True)
        if len(teams) < 2:
            continue

        home_abbr = teams['_home_abbr'].iloc[0]
        tid_upper = teams['team_id'].str.upper()
        home_mask = (tid_upper == home_abbr) | tid_upper.str.contains(home_abbr, na=False, regex=False)

        if home_mask.sum() == 1:
            home_row = teams[home_mask].iloc[0]
            away_row = teams[~home_mask].iloc[0]
        else:
            home_row, away_row = teams.iloc[0], teams.iloc[1]

        record = {
            'game_id': game_id,
            'hometeam': home_row['team_id'],
            'awayteam': away_row['team_id'],
            'homescore': int(home_row['score']),
            'awayscore': int(away_row['score']),
        }
        for col in ('phase', 'season', 'round'):
            if col in teams.columns:
                record[col] = teams[col].iloc[0]
        records.append(record)

    result_df = pd.DataFrame(records)
    print(f"  {len(df):,} player rows -> {len(result_df):,} game rows")
    return result_df


def encode_categoricals(df):
    """Label-encode hometeam, awayteam, phase and save the fitted encoders."""
    encoders = {}
    for col in ['hometeam', 'awayteam', 'phase']:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} unique values")
    return df, encoders


def build_feature_list(df):
    """Return the ordered list of feature column names used by the models."""
    candidates = [
        'hometeam_encoded', 'awayteam_encoded', 'phase_encoded',
        'year', 'home_form', 'away_form', 'head_to_head_advantage',
    ]
    return [c for c in candidates if c in df.columns]


def run_pipeline(input_filepath=None):
    """End-to-end data preparation."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data(input_filepath)
    df = normalize_columns(df)

    explore_data(df)

    # If the CSV is a player-level box score, aggregate to game level first
    if 'player_id' in df.columns or 'game_player_id' in df.columns:
        df = aggregate_box_score_to_games(df)

    # Parse dates early — required for temporal feature engineering
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        before = len(df)
        df = df.dropna(subset=['date'])
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with unparseable dates.")
    else:
        # If no date column, create a proxy from season/round ordering
        sort_cols = [c for c in ('season', 'round') if c in df.columns] or [df.columns[0]]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df['date'] = pd.date_range(start='2007-01-01', periods=len(df), freq='3D')
        print("Warning: no 'date' column found. Assigned synthetic dates.")

    df = normalize_phases(df)
    df = handle_missing_values(df)

    print("\n=== Preparing result column ===")
    df = prepare_result_column(df)

    print("\n=== Extracting year ===")
    df = extract_year(df)

    print("\n=== Computing team forms ===")
    df = compute_team_forms(df)

    print("\n=== Computing head-to-head advantage ===")
    df = compute_head_to_head(df)

    print("\n=== Encoding categorical features ===")
    df, encoders = encode_categoricals(df)

    with open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)

    feature_cols = build_feature_list(df)
    print(f"\nFinal features: {feature_cols}")
    print(f"Target        : result")

    metadata = {
        'feature_cols': feature_cols,
        'target_col': 'result',
    }
    with open(os.path.join(DATA_PROCESSED_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    out_path = os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved processed data -> {out_path}  (shape: {df.shape})")

    return df, encoders, feature_cols


if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(filepath)
