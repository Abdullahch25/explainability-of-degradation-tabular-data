"""Compute per-fold CV metrics for all 5 models used for error bars in Fig 13.1."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

meta_df = pd.read_csv('step11_meta_features.csv')
y = meta_df['accuracy_drop'].values

CT_COLS = sorted(c for c in meta_df.columns if c.startswith('ct_'))
FULL_FEATS = [
    'mean_tier_shap', 'severity', 'shap_x_severity',
    'mean_ks_stat', 'max_ks_stat', 'drift_pct_scipy',
    'tier_num', 'log_n_instances', 'n_features_out',
    'n_classes', 'imbalanced_int', 'baseline_accuracy',
] + CT_COLS

meta_df['severity_sq'] = meta_df['severity'] ** 2

X_sev      = meta_df[['severity']].values
X_sev_quad = meta_df[['severity', 'severity_sq']].values
X_heur     = meta_df[['shap_x_severity']].values
X_full     = meta_df[FULL_FEATS].values

models = [
    ('Severity-Only Baseline',             LinearRegression(),                                              X_sev),
    ('Severity + Severity² Baseline',       LinearRegression(),                                              X_sev_quad),
    ('Heuristic (shap_x_severity only)',    LinearRegression(),                                              X_heur),
    ('Linear Regression (full features)',   LinearRegression(),                                              X_full),
    ('Random Forest Regressor',             RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), X_full),
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for model_name, model, X in models:
    fold_rmse, fold_r2, fold_r = [], [], []
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], y_pred))))
        fold_r2.append(float(r2_score(y[test_idx], y_pred)))
        r_val, _ = pearsonr(y[test_idx], y_pred)
        fold_r.append(float(r_val))
    rows.append({
        'model':          model_name,
        'rmse_mean':      np.mean(fold_rmse),  'rmse_std':      np.std(fold_rmse),
        'r2_mean':        np.mean(fold_r2),    'r2_std':        np.std(fold_r2),
        'pearson_r_mean': np.mean(fold_r),     'pearson_r_std': np.std(fold_r),
    })
    print(f"  {model_name:45s}  RMSE={np.mean(fold_rmse):.5f}±{np.std(fold_rmse):.5f}"
          f"  R²={np.mean(fold_r2):.4f}±{np.std(fold_r2):.4f}"
          f"  r={np.mean(fold_r):.4f}±{np.std(fold_r):.4f}")

pd.DataFrame(rows).to_csv('step13_fold_metrics.csv', index=False)
