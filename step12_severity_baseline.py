import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

#Load meta-features table (already assembled in Step 11)
meta_df = pd.read_csv('step11_meta_features.csv')
y = meta_df['accuracy_drop'].values
print(f"Loaded {len(meta_df)} experiments")

#Feature sets for baselines
#severity only (raw)
X_sev      = meta_df[['severity']].values

# severity + severity^2 (non-linear extension)
meta_df['severity_sq'] = meta_df['severity'] ** 2
X_sev_quad = meta_df[['severity', 'severity_sq']].values

#Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    print(f"  {name:45s}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}  r={float(r):.4f}")
    return {
        'model':      name,
        'rmse':       rmse,
        'mae':        mae,
        'r2':         r2,
        'pearson_r':  float(r),
        'pearson_p':  float(p),
    }

print("\n5-fold cross-validation (severity baselines):")
metrics_rows = []

sev_model      = LinearRegression()
y_pred_sev     = cross_val_predict(sev_model, X_sev, y, cv=kf)
metrics_rows.append(cv_metrics(y, y_pred_sev, 'Severity-Only Baseline'))

sev_quad_model  = LinearRegression()
y_pred_sev_quad = cross_val_predict(sev_quad_model, X_sev_quad, y, cv=kf)
metrics_rows.append(cv_metrics(y, y_pred_sev_quad, 'Severity + Severity² Baseline'))

#Fit final models on full data (for coefficient inspection)
sev_model.fit(X_sev, y)
sev_quad_model.fit(X_sev_quad, y)

print(f"\nSeverity-Only coefficient    : {sev_model.coef_[0]:.6f}  "
      f"intercept: {sev_model.intercept_:.6f}")
print(f"Severity²  coefficients      : sev={sev_quad_model.coef_[0]:.6f}  "
      f"sev²={sev_quad_model.coef_[1]:.6f}  "
      f"intercept: {sev_quad_model.intercept_:.6f}")

#Save outputs
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('step12_evaluation.csv', index=False)

preds_df = meta_df[['dataset', 'tier', 'corruption_type', 'severity', 'accuracy_drop']].copy()
preds_df['pred_severity_only'] = y_pred_sev
preds_df['pred_severity_quad'] = y_pred_sev_quad
preds_df.to_csv('step12_predictions.csv', index=False)

print(f"\n{'='*60}")
print("Saved:")
print("  step12_evaluation.csv   — CV metrics for severity baselines")
print("  step12_predictions.csv  — actual vs predicted")
