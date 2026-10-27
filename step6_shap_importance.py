import pandas as pd
import numpy as np
import os
import pickle
import shap

PROCESSED_DIR = 'data/processed'
META_DIR      = 'data/preprocessors'
MODELS_DIR    = 'models'
SHAP_DIR      = 'shap'
os.makedirs(SHAP_DIR, exist_ok=True)

load_info = pd.read_csv('step2_load_summary.csv')
load_info = load_info[load_info['status'] == 'ok']

# Cap test samples used for SHAP to keep computation manageable
SHAP_SAMPLE_CAP = 500
RANDOM_STATE    = 42

summary_rows = []

for _, row in load_info.iterrows():
    name = row['name']
    print(f"--- {name} ---")

    test_df = pd.read_csv(f"{PROCESSED_DIR}/{name}_test.csv")
    X_test  = test_df.drop(columns=['target']).values

    with open(f"{META_DIR}/{name}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)

    with open(f"{MODELS_DIR}/{name}_rf.pkl", 'rb') as f:
        rf = pickle.load(f)

    feature_names = meta['feature_names']
    n_classes     = len(meta['classes'])

    # Subsample for large test sets
    if len(X_test) > SHAP_SAMPLE_CAP:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_test), size=SHAP_SAMPLE_CAP, replace=False)
        X_shap = X_test[idx]
        print(f"  Subsampled : {SHAP_SAMPLE_CAP}/{len(X_test)} test rows for SHAP")
    else:
        X_shap = X_test
        print(f"  SHAP rows  : {len(X_shap)}")

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)   # (n_samples, n_features, n_classes)
    else:
        shap_array = np.array(shap_values)            # (n_samples, n_features, n_classes)
    importance = np.mean(np.abs(shap_array), axis=(0, 2))  # (n_features,)

    imp_df = pd.DataFrame({
        'feature':          feature_names,
        'shap_importance':  importance,
    })
    imp_df = imp_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    imp_df['rank'] = imp_df.index + 1

    out_path = f"{SHAP_DIR}/{name}_shap_importance.csv"
    imp_df.to_csv(out_path, index=False)

    top5 = imp_df.head(5)['feature'].tolist()
    print(f"  Top 5 features: {top5}")
    print(f"  Saved: {out_path}\n")

    summary_rows.append({
        'name':       name,
        'n_features': len(feature_names),
        'n_shap_rows': len(X_shap),
        'top1':       imp_df.iloc[0]['feature'],
        'top2':       imp_df.iloc[1]['feature'] if len(imp_df) > 1 else '',
        'top3':       imp_df.iloc[2]['feature'] if len(imp_df) > 2 else '',
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv('step6_shap_summary.csv', index=False)

print("=" * 60)
print("SHAP feature importance complete.")
print(f"SHAP files saved : {SHAP_DIR}/")
print(summary[['name', 'n_features', 'top1', 'top2', 'top3']].to_string())
