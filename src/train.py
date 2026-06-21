"""
Model training with GridSearchCV hyperparameter tuning.
Trains: Logistic Regression, Decision Tree, Random Forest, XGBoost.
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
import time

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')

MODELS_CONFIG = {
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42,
                                    class_weight={0: 1.4, 1: 1.0}),
        'params': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear'],
        },
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(random_state=42,
                                        class_weight={0: 1.4, 1: 1.0}),
        'params': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1,
                                        class_weight={0: 1.4, 1: 1.0}),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        },
    },
    'xgboost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss',
                               verbosity=0, use_label_encoder=False,
                               scale_pos_weight=0.75),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        },
    },
}


def load_processed_data():
    data_path = os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv')
    meta_path = os.path.join(DATA_PROCESSED_DIR, 'metadata.json')

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run data_preparation.py first."
        )

    df = pd.read_csv(data_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    return df, metadata


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        'accuracy':  float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall':    float(recall_score(y, y_pred, zero_division=0)),
        'f1':        float(f1_score(y, y_pred, zero_division=0)),
        'f1_macro':  float(f1_score(y, y_pred, average='macro', zero_division=0)),
    }


def train_all_models(df, feature_cols, target_col='result'):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Temporal split: sort chronologically, train on older seasons, test on newest ~20%.
    # Random split would mix seasons, letting the model "see" future games during training.
    sort_col = 'year' if 'year' in df.columns else df.columns[0]
    df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)

    X = df_sorted[feature_cols].fillna(0)
    y = df_sorted[target_col]

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    if 'year' in df_sorted.columns:
        train_max = df_sorted.iloc[:split_idx]['year'].max()
        test_yrs  = sorted(df_sorted.iloc[split_idx:]['year'].unique().tolist())
        print(f"Train: {len(X_train)} games (up to {train_max})  |  "
              f"Test: {len(X_test)} games (seasons: {test_yrs})")
    else:
        print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Scaler only used by Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Persist test set for evaluate.py
    X_test.to_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'))
    y_test.to_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv'))

    # TimeSeriesSplit respects temporal order within training data (no future leakage in CV)
    cv = TimeSeriesSplit(n_splits=5)
    all_results = {}

    for name, cfg in MODELS_CONFIG.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print('='*55)

        uses_scaler = (name == 'logistic_regression')
        Xtr = X_train_sc if uses_scaler else X_train
        Xte = X_test_sc  if uses_scaler else X_test

        t0 = time.time()
        gs = GridSearchCV(
            cfg['model'], cfg['params'],
            cv=cv, scoring='balanced_accuracy',
            n_jobs=-1, verbose=0,
        )
        gs.fit(Xtr, y_train)
        elapsed = time.time() - t0

        best = gs.best_estimator_
        metrics = evaluate_model(best, Xte, y_test)
        metrics['best_params']   = gs.best_params_
        metrics['best_cv_score'] = float(gs.best_score_)
        metrics['training_time'] = round(elapsed, 1)

        print(f"  Best params : {gs.best_params_}")
        print(f"  CV accuracy : {gs.best_score_:.4f}")
        print(f"  Test accuracy : {metrics['accuracy']:.4f}  "
              f"Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}  "
              f"F1: {metrics['f1']:.4f}")
        print(f"  Training time: {elapsed:.1f}s")

        with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'wb') as f:
            pickle.dump(best, f)

        all_results[name] = metrics

    with open(os.path.join(METRICS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    best_name = max(all_results, key=lambda k: all_results[k]['f1_macro'])
    with open(os.path.join(MODELS_DIR, 'best_model.json'), 'w') as f:
        json.dump({'best_model': best_name}, f)

    print(f"\n{'='*55}")
    print(f"  Best overall model: {best_name}  (F1={all_results[best_name]['f1']:.4f})")
    print('='*55)

    return all_results


def main():
    print("Loading processed data...")
    df, metadata = load_processed_data()

    feature_cols = metadata['feature_cols']
    target_col   = metadata['target_col']

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Samples : {len(df)}")
    print(f"Class balance:\n{df[target_col].value_counts()}")

    results = train_all_models(df, feature_cols, target_col)

    print("\n" + "="*65)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print('-'*65)
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")


if __name__ == '__main__':
    main()
