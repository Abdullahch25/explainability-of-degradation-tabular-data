import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load all CV metrics
s11 = pd.read_csv('step11_evaluation.csv')
s12 = pd.read_csv('step12_evaluation.csv')

all_metrics = pd.concat([s12, s11], ignore_index=True)

#Canonical order for display
MODEL_ORDER = [
    'Severity-Only Baseline',
    'Severity + Severity² Baseline',
    'Heuristic (shap_x_severity only)',
    'Linear Regression (full features)',
    'Random Forest Regressor',
]
all_metrics['_order'] = all_metrics['model'].map({m: i for i, m in enumerate(MODEL_ORDER)})
all_metrics = all_metrics.sort_values('_order').drop(columns='_order').reset_index(drop=True)
all_metrics.to_csv('step13_all_metrics.csv', index=False)

#Improvement over severity-only baseline
baseline_rmse = all_metrics.loc[all_metrics['model'] == 'Severity-Only Baseline', 'rmse'].iloc[0]
baseline_r2   = all_metrics.loc[all_metrics['model'] == 'Severity-Only Baseline', 'r2'].iloc[0]

all_metrics['rmse_improvement_pct'] = ((baseline_rmse - all_metrics['rmse']) / baseline_rmse * 100).round(1)
all_metrics['r2_gain']              = (all_metrics['r2'] - baseline_r2).round(4)

#Load predictions for residual analysis
meta_df  = pd.read_csv('step11_meta_features.csv')
preds11  = pd.read_csv('step11_predictions.csv')
preds12  = pd.read_csv('step12_predictions.csv')


combined = meta_df[['dataset', 'tier', 'corruption_type', 'severity', 'accuracy_drop']].copy()
combined = combined.merge(
    preds12[['dataset', 'tier', 'corruption_type', 'severity',
              'pred_severity_only', 'pred_severity_quad']],
    on=['dataset', 'tier', 'corruption_type', 'severity'])
combined = combined.merge(
    preds11[['dataset', 'tier', 'corruption_type', 'severity',
              'pred_heuristic', 'pred_lr', 'pred_rf']],
    on=['dataset', 'tier', 'corruption_type', 'severity'])

combined.to_csv('step13_all_predictions.csv', index=False)

# Per-tier and per-corruption breakdown for best model (RF) 
def breakdown_metrics(df, group_col, pred_col, true_col='accuracy_drop'):
    rows = []
    for val, grp in df.groupby(group_col):
        y_t = grp[true_col].values
        y_p = grp[pred_col].values
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        r2   = float(1 - np.sum((y_t - y_p)**2) / np.sum((y_t - y_t.mean())**2))
        r, _ = pearsonr(y_t, y_p)
        rows.append({group_col: val, 'rmse': round(rmse, 5),
                     'r2': round(r2, 4), 'pearson_r': round(float(r), 4), 'n': len(grp)})
    return pd.DataFrame(rows)

rf_by_tier       = breakdown_metrics(combined, 'tier',            'pred_rf')
rf_by_corruption = breakdown_metrics(combined, 'corruption_type', 'pred_rf')
rf_by_severity   = breakdown_metrics(combined, 'severity',        'pred_rf')

rf_by_tier.to_csv('step13_rf_by_tier.csv', index=False)
rf_by_corruption.to_csv('step13_rf_by_corruption.csv', index=False)
rf_by_severity.to_csv('step13_rf_by_severity.csv', index=False)

#print final report 
SEP = '=' * 72

print(SEP)
print("STEP 13 — FINAL EVALUATION & COMPARISON")
print(SEP)

print("\n── All Models: 5-fold CV Performance ──────────────────────────────────")
display = all_metrics[['model', 'rmse', 'mae', 'r2', 'pearson_r',
                         'rmse_improvement_pct', 'r2_gain']].copy()
display.columns = ['Model', 'RMSE', 'MAE', 'R²', 'r', 'RMSE↓ vs sev%', 'R²↑ vs sev']
print(display.to_string(index=False))

print("\n── RF Regressor: Performance by Tier ──────────────────────────────────")
print(rf_by_tier.to_string(index=False))

print("\n── RF Regressor: Performance by Corruption Type ───────────────────────")
print(rf_by_corruption.sort_values('rmse').to_string(index=False))

print("\n── RF Regressor: Performance by Severity ──────────────────────────────")
print(rf_by_severity.to_string(index=False))

print(f"\n{SEP}")
print("CONCLUSIONS")
print(SEP)

rf_row    = all_metrics[all_metrics['model'] == 'Random Forest Regressor'].iloc[0]
sev_row   = all_metrics[all_metrics['model'] == 'Severity-Only Baseline'].iloc[0]
heur_row  = all_metrics[all_metrics['model'] == 'Heuristic (shap_x_severity only)'].iloc[0]
lr_row    = all_metrics[all_metrics['model'] == 'Linear Regression (full features)'].iloc[0]

print(f"\n1. SHAP tier adds significant value over severity alone:")
print(f"   Severity-only  R²={sev_row['r2']:.4f}  →  Heuristic R²={heur_row['r2']:.4f}"
      f"  (+{heur_row['r2_gain']:.4f}, +{heur_row['rmse_improvement_pct']:.1f}% RMSE)")

print(f"\n2. Full feature set (LR) improves further:")
print(f"   Heuristic R²={heur_row['r2']:.4f}  →  LR R²={lr_row['r2']:.4f}"
      f"  (+{lr_row['r2_gain'] - heur_row['r2_gain']:.4f})")

print(f"\n3. Non-linear interactions captured by RF give best performance:")
print(f"   LR R²={lr_row['r2']:.4f}  →  RF R²={rf_row['r2']:.4f}"
      f"  (+{rf_row['r2_gain'] - lr_row['r2_gain']:.4f},"
      f" {rf_row['rmse_improvement_pct']:.1f}% RMSE reduction vs severity baseline)")

print(f"\n4. Severity alone explains {sev_row['r2']*100:.1f}% of variance in accuracy drop.")
print(f"   SHAP × severity (heuristic) explains {heur_row['r2']*100:.1f}% — "
      f"{((heur_row['r2']-sev_row['r2'])/sev_row['r2']*100):.0f}% relative gain.")
print(f"   RF with full features explains {rf_row['r2']*100:.1f}% — "
      f"{((rf_row['r2']-sev_row['r2'])/sev_row['r2']*100):.0f}% relative gain over severity.")

print(f"\n{SEP}")
print("Saved:")
print("  step13_all_metrics.csv       — combined CV metrics all models")
print("  step13_all_predictions.csv   — all model predictions per experiment")
print("  step13_rf_by_tier.csv        — RF breakdown by SHAP tier")
print("  step13_rf_by_corruption.csv  — RF breakdown by corruption type")
print("  step13_rf_by_severity.csv    — RF breakdown by severity level")
