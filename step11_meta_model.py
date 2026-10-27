import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

#Load inputs
res_df   = pd.read_csv('step9_performance_results.csv')
drift_df = pd.read_csv('step10_drift_results.csv')
ds_df    = pd.read_csv('step1_selected_datasets.csv')
pre_df   = pd.read_csv('step3_preprocessing_summary.csv')
base_df  = pd.read_csv('step5_baseline_evaluation.csv')
plan_df  = pd.read_csv('step7_corruption_plan.csv')

#Compute mean SHAP importance per (dataset, tier)
print("Computing SHAP importance per (dataset, tier)...")
tier_feat = (plan_df[['dataset', 'tier', 'features']]
             .drop_duplicates(subset=['dataset', 'tier']))
shap_rows = []
for _, row in tier_feat.iterrows():
    name    = row['dataset']
    tier    = row['tier']
    feats   = row['features'].split('|')
    sd      = pd.read_csv(f'shap/{name}_shap_importance.csv')
    matched = sd[sd['feature'].isin(feats)]['shap_importance']
    if len(matched):
        shap_rows.append({
            'dataset':        name,
            'tier':           tier,
            'mean_tier_shap': float(matched.mean()),
            'max_tier_shap':  float(matched.max()),
        })
shap_tier_df = pd.DataFrame(shap_rows)
print(f"  {len(shap_tier_df)} (dataset, tier) pairs")

#Assemble meta-feature table
print("\nAssembling meta-feature table...")

meta_df = res_df[['dataset', 'tier', 'corruption_type', 'severity',
                   'accuracy_drop', 'f1_drop', 'auc_drop']].copy()

meta_df = meta_df.merge(
    drift_df[['dataset', 'tier', 'corruption_type', 'severity',
               'mean_ks_stat', 'max_ks_stat', 'drift_pct_scipy']],
    on=['dataset', 'tier', 'corruption_type', 'severity'], how='left')

meta_df = meta_df.merge(shap_tier_df, on=['dataset', 'tier'], how='left')

ds_meta = (ds_df[['name', 'n_instances', 'n_classes']]
           .rename(columns={'name': 'dataset'}))
ds_meta = ds_meta.merge(
    pre_df[['name', 'n_features_out', 'imbalanced', 'minority_pct']]
    .rename(columns={'name': 'dataset'}),
    on='dataset')
ds_meta = ds_meta.merge(
    base_df[['name', 'accuracy']]
    .rename(columns={'name': 'dataset', 'accuracy': 'baseline_accuracy'}),
    on='dataset')
meta_df = meta_df.merge(ds_meta, on='dataset', how='left')

#Derived features
tier_map = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
meta_df['tier_num']        = meta_df['tier'].map(tier_map)
meta_df['shap_x_severity'] = meta_df['mean_tier_shap'] * meta_df['severity']
meta_df['log_n_instances'] = np.log10(meta_df['n_instances'])
meta_df['imbalanced_int']  = meta_df['imbalanced'].astype(int)

#One-hot encode corruption type (keep all 7 with drop_first=False for interpretability)
ct_dummies = pd.get_dummies(meta_df['corruption_type'], prefix='ct')
meta_df    = pd.concat([meta_df, ct_dummies], axis=1)

meta_df.to_csv('step11_meta_features.csv', index=False)
print(f"  Shape: {meta_df.shape[0]} rows x {meta_df.shape[1]} columns")

#Define feature sets and target
TARGET = 'accuracy_drop'
y = meta_df[TARGET].values

HEURISTIC_FEATS = ['shap_x_severity']

CT_COLS    = sorted(c for c in meta_df.columns if c.startswith('ct_'))
FULL_FEATS = [
    'mean_tier_shap', 'severity', 'shap_x_severity',
    'mean_ks_stat', 'max_ks_stat', 'drift_pct_scipy',
    'tier_num', 'log_n_instances', 'n_features_out',
    'n_classes', 'imbalanced_int', 'baseline_accuracy',
] + CT_COLS

X_heuristic = meta_df[HEURISTIC_FEATS].values
X_full      = meta_df[FULL_FEATS].values

print(f"\nFeature sets:")
print(f"  Heuristic : {HEURISTIC_FEATS}")
print(f"  Full      : {len(FULL_FEATS)} features")

#Cross-validation (5-fold)
def cv_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    print(f"  {name:40s}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}  r={float(r):.4f}")
    return {'model': name, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'pearson_r': float(r), 'pearson_p': float(p)}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n5-fold cross-validation:")

metrics_rows = []

#Heuristic: shap_x_severity only
heur_model  = LinearRegression()
y_pred_heur = cross_val_predict(heur_model, X_heuristic, y, cv=kf)
metrics_rows.append(cv_metrics(y, y_pred_heur, 'Heuristic (shap_x_severity only)'))

#Linear regression: full feature set
lr_model  = LinearRegression()
y_pred_lr = cross_val_predict(lr_model, X_full, y, cv=kf)
metrics_rows.append(cv_metrics(y, y_pred_lr, 'Linear Regression (full features)'))

#Random Forest: full feature set
rf_model  = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
y_pred_rf = cross_val_predict(rf_model, X_full, y, cv=kf)
metrics_rows.append(cv_metrics(y, y_pred_rf, 'Random Forest Regressor'))

#fit final models on all 807 experiments
print("\nFitting final models on full dataset...")
heur_model.fit(X_heuristic, y)
lr_model.fit(X_full, y)
rf_model.fit(X_full, y)

#Diagnostics
feat_imp = pd.DataFrame({
    'feature':    FULL_FEATS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

lr_coef = pd.DataFrame({
    'feature':     FULL_FEATS,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', ascending=False, key=abs)

print("\nTop 10 RF feature importances:")
print(feat_imp.head(10).to_string(index=False))

print("\nTop 10 LR coefficients (by |coef|):")
print(lr_coef.head(10).to_string(index=False))

print(f"\nHeuristic coefficient (shap_x_severity -> accuracy_drop): "
      f"{heur_model.coef_[0]:.4f}  intercept: {heur_model.intercept_:.4f}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('step11_evaluation.csv', index=False)

preds_df = meta_df[['dataset', 'tier', 'corruption_type',
                     'severity', 'accuracy_drop']].copy()
preds_df['pred_heuristic'] = y_pred_heur
preds_df['pred_lr']        = y_pred_lr
preds_df['pred_rf']        = y_pred_rf
preds_df.to_csv('step11_predictions.csv', index=False)

feat_imp.to_csv('step11_rf_feature_importance.csv', index=False)
lr_coef.to_csv('step11_lr_coefficients.csv', index=False)

models_bundle = {
    'heuristic':          heur_model,
    'linear_regression':  lr_model,
    'random_forest':      rf_model,
    'features_heuristic': HEURISTIC_FEATS,
    'features_full':      FULL_FEATS,
}
with open('step11_models.pkl', 'wb') as f:
    pickle.dump(models_bundle, f)

print(f"\n{'='*60}")
print("Saved:")
print("  step11_meta_features.csv        — 807-row feature table")
print("  step11_evaluation.csv           — CV metrics per model")
print("  step11_predictions.csv          — actual vs predicted")
print("  step11_rf_feature_importance.csv")
print("  step11_lr_coefficients.csv")
print("  step11_models.pkl               — fitted model objects")
