"""
Model evaluation: metrics, confusion matrices, feature importance,
distribution plots, season accuracy, feature selection comparison.
"""
import pandas as pd
import numpy as np
import os
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR        = os.path.join(BASE_DIR, 'models')
FIGURES_DIR       = os.path.join(BASE_DIR, 'results', 'figures')
METRICS_DIR       = os.path.join(BASE_DIR, 'results', 'metrics')

DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'xgboost':             'XGBoost',
}

FEATURE_LABELS = {
    'hometeam_encoded':      'Home Team',
    'awayteam_encoded':      'Away Team',
    'phase_encoded':         'Phase',
    'year':                  'Year',
    'home_form':             'Home Form',
    'away_form':             'Away Form',
    'head_to_head_advantage':'H2H Advantage',
}


def _label(feat):
    return FEATURE_LABELS.get(feat, feat)


def load_artifacts():
    with open(os.path.join(DATA_PROCESSED_DIR, 'metadata.json')) as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']

    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv'), index_col=0).squeeze()

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    models = {}
    for name in DISPLAY:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    df_full = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'processed_data.csv'))

    return models, X_test, X_test_sc, y_test, feature_cols, scaler, df_full


def _get_X(name, X_test, X_test_sc):
    return X_test_sc if name == 'logistic_regression' else X_test


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(models, X_test, X_test_sc, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(_get_X(name, X_test, X_test_sc))
        cm  = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Away Win', 'Home Win'],
                    yticklabels=['Away Win', 'Home Win'],
                    ax=axes[i])
        axes[i].set_title(f'{DISPLAY[name]}\n(Accuracy: {acc:.3f})')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('confusion_matrices.png')


def plot_metrics_comparison(models, X_test, X_test_sc, y_test):
    rows = []
    for name, model in models.items():
        y_pred = model.predict(_get_X(name, X_test, X_test_sc))
        rows.append({
            'Model':     DISPLAY[name],
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1':        f1_score(y_test, y_pred, zero_division=0),
        })

    mdf = pd.DataFrame(rows).set_index('Model')
    mdf.to_csv(os.path.join(METRICS_DIR, 'all_metrics.csv'))

    x = np.arange(len(mdf))
    w = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, col in enumerate(mdf.columns):
        bars = ax.bar(x + j * w, mdf[col], w, label=col, color=colors[j], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(mdf.index, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save('metrics_comparison.png')
    return mdf


def plot_feature_importance(models, X_test, X_test_sc, y_test, feature_cols):
    labels = [_label(f) for f in feature_cols]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]

        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            title = 'Feature Importance'
        elif hasattr(model, 'coef_'):
            imps = np.abs(model.coef_[0])
            title = '|Coefficient|'
        else:
            imps = np.ones(len(feature_cols))
            title = 'N/A'

        order = np.argsort(imps)
        colors = plt.cm.RdYlGn(imps[order] / (imps.max() + 1e-9))
        ax.barh([labels[j] for j in order], imps[order], color=colors)
        ax.set_title(f'{DISPLAY[name]}\n{title}')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('feature_importance.png')


def plot_outcome_distribution(df):
    if 'result' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    counts = df['result'].value_counts()
    lbls   = ['Home Win' if v == 1 else 'Away Win' for v in counts.index]
    axes[0].pie(counts.values, labels=lbls,
                colors=['#2196F3', '#FF5722'],
                autopct='%1.1f%%', startangle=90, shadow=True)
    axes[0].set_title('Overall Outcome Distribution')

    # Home win rate by phase
    if 'phase' in df.columns and df['phase'].nunique() <= 15:
        phase_rate = df.groupby('phase')['result'].mean().sort_values(ascending=False)
        bars = axes[1].bar(range(len(phase_rate)), phase_rate.values,
                           color=plt.cm.RdYlGn(phase_rate.values))
        axes[1].set_xticks(range(len(phase_rate)))
        axes[1].set_xticklabels(phase_rate.index, rotation=40, ha='right')
        axes[1].set_ylabel('Home Win Rate')
        axes[1].set_title('Home Win Rate by Phase')
        axes[1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylim(0, 1)
        for bar, val in zip(bars, phase_rate.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _save('outcome_distribution.png')


def plot_score_distribution(df):
    if 'homescore' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(df['homescore'].dropna(), bins=30, alpha=0.7,
                 label='Home Score', color='#2196F3', edgecolor='white')
    axes[0].hist(df['awayscore'].dropna(), bins=30, alpha=0.7,
                 label='Away Score', color='#FF5722', edgecolor='white')
    axes[0].set_xlabel('Points')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Score Distribution: Home vs Away')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    diff = df['homescore'] - df['awayscore']
    axes[1].hist(diff.dropna(), bins=40, color='#4CAF50', alpha=0.85, edgecolor='black', lw=0.4)
    axes[1].axvline(0, color='red', linestyle='--', label='Even score')
    axes[1].set_xlabel('Home Score − Away Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Difference Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save('score_distribution.png')


def plot_accuracy_by_season(models, df, feature_cols, scaler):
    if 'year' not in df.columns:
        return

    years = sorted(df['year'].dropna().unique())
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = {
        'logistic_regression': '#2196F3',
        'decision_tree':       '#FF9800',
        'random_forest':       '#4CAF50',
        'xgboost':             '#E91E63',
    }

    for name, model in models.items():
        accs, valid_years = [], []
        for yr in years:
            sub = df[df['year'] == yr]
            if len(sub) < 5:
                continue
            X_s = sub[feature_cols].fillna(0)
            y_s = sub['result']
            if name == 'logistic_regression':
                X_s = pd.DataFrame(scaler.transform(X_s), columns=X_s.columns)
            y_pred = model.predict(X_s)
            accs.append(accuracy_score(y_s, y_pred))
            valid_years.append(yr)

        if valid_years:
            ax.plot(valid_years, accs, marker='o',
                    label=DISPLAY[name], color=palette.get(name), lw=2)

    ax.axhline(0.5, color='black', linestyle='--', alpha=0.4, label='Random baseline')
    ax.set_xlabel('Season Start Year')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy per Season')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _save('accuracy_by_season.png')


def feature_selection_comparison(models, X_test, X_test_sc, y_test, feature_cols, df_full):
    """
    Compare accuracy using ALL features vs. only the top half (by RF importance).
    Retrains Random Forest from scratch on top features for a fair comparison.
    """
    from sklearn.ensemble import RandomForestClassifier

    if 'random_forest' not in models:
        return

    rf   = models['random_forest']
    imps = rf.feature_importances_
    top_k = max(3, len(feature_cols) // 2)
    top_idx = np.argsort(imps)[-top_k:]
    top_feats = [feature_cols[i] for i in top_idx]

    print(f"\nTop {top_k} features: {[_label(f) for f in top_feats]}")

    # Rebuild the same 80/20 temporal split used during training
    sort_col = 'year' if 'year' in df_full.columns else df_full.columns[0]
    df_sorted = df_full.sort_values(sort_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    X_train_all = df_sorted[feature_cols].fillna(0).iloc[:split_idx]
    y_train     = df_sorted['result'].iloc[:split_idx]

    # Retrain RF with the same hyperparameters but only on top features
    rf_top = RandomForestClassifier(**rf.get_params())
    rf_top.fit(X_train_all[top_feats], y_train)

    acc_all = accuracy_score(y_test, rf.predict(X_test))
    acc_top = accuracy_score(y_test, rf_top.predict(X_test[top_feats]))

    print(f"  RF (svi atributi, n={len(feature_cols)}): accuracy = {acc_all:.4f}")
    print(f"  RF (top atributi, n={top_k}):            accuracy = {acc_top:.4f}")

    results = {
        'all_features': {
            'n_features': len(feature_cols),
            'features':   [_label(f) for f in feature_cols],
            'accuracy':   round(acc_all, 4),
        },
        'top_features': {
            'n_features': top_k,
            'features':   [_label(f) for f in top_feats],
            'accuracy':   round(acc_top, 4),
        },
    }

    out = os.path.join(METRICS_DIR, 'feature_selection.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved feature selection analysis -> {out}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: RF feature importance bar chart
    order  = np.argsort(imps)
    colors = ['#E91E63' if i in top_idx else '#90CAF9' for i in order]
    axes[0].barh([_label(feature_cols[i]) for i in order], imps[order], color=colors)
    axes[0].set_title('Random Forest Feature Importance\n(pink = top features)')
    axes[0].set_xlabel('Importance')
    axes[0].grid(axis='x', alpha=0.3)

    # Right: accuracy comparison all vs top features
    labels_bar = [f'Svi atributi\n(n={len(feature_cols)})', f'Top atributi\n(n={top_k})']
    accs = [acc_all, acc_top]
    bars = axes[1].bar(labels_bar, accs, color=['#4CAF50', '#FF9800'], alpha=0.85, width=0.4)
    for bar, val in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Accuracy: Svi vs Top atributi\n(Random Forest)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save('feature_selection_rf.png')

    return results, top_feats


def plot_correlation_matrix(df, feature_cols):
    cols = [c for c in feature_cols if c in df.columns] + ['result']
    labels = [_label(c) if c != 'result' else 'Result' for c in cols]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels,
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Korelaciona matrica atributa', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save('correlation_matrix.png')


def print_classification_reports(models, X_test, X_test_sc, y_test):
    out_path = os.path.join(METRICS_DIR, 'classification_reports.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for name, model in models.items():
            y_pred = model.predict(_get_X(name, X_test, X_test_sc))
            report = classification_report(
                y_test, y_pred,
                target_names=['Away Win', 'Home Win'],
            )
            block = f"\n{'='*50}\n{DISPLAY[name]}\n{'='*50}\n{report}\n"
            print(block)
            f.write(block)
    print(f"Saved reports -> {out_path}")


def _save(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading models and data...")
    models, X_test, X_test_sc, y_test, feature_cols, scaler, df_full = load_artifacts()

    if not models:
        raise RuntimeError("No trained models found. Run train.py first.")

    print("\n=== Classification Reports ===")
    print_classification_reports(models, X_test, X_test_sc, y_test)

    print("\n=== Confusion Matrices ===")
    plot_confusion_matrices(models, X_test, X_test_sc, y_test)

    print("\n=== Metrics Comparison ===")
    mdf = plot_metrics_comparison(models, X_test, X_test_sc, y_test)

    print("\n=== Feature Importance ===")
    plot_feature_importance(models, X_test, X_test_sc, y_test, feature_cols)

    print("\n=== Outcome Distribution ===")
    plot_outcome_distribution(df_full)

    print("\n=== Score Distribution ===")
    plot_score_distribution(df_full)

    print("\n=== Accuracy by Season ===")
    plot_accuracy_by_season(models, df_full, feature_cols, scaler)

    print("\n=== Correlation Matrix ===")
    plot_correlation_matrix(df_full, feature_cols)

    print("\n=== Feature Selection Comparison ===")
    feature_selection_comparison(models, X_test, X_test_sc, y_test, feature_cols, df_full)

    print("\n=== Summary ===")
    print(mdf.round(4).to_string())
    print(f"\nFigures -> {FIGURES_DIR}")
    print(f"Metrics -> {METRICS_DIR}")


if __name__ == '__main__':
    main()
