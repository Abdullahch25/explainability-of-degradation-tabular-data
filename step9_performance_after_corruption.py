import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROCESSED_DIR     = 'data/processed'
CORRUPTED_DIR     = 'data/corrupted'
META_DIR          = 'data/preprocessors'
MODELS_DIR        = 'models'
P_VALUE_THRESHOLD = 0.05

manifest = pd.read_csv('step8_manifest.csv')
baseline = pd.read_csv('step5_baseline_evaluation.csv').set_index('name')

# Pre-load models, meta and clean predictions once per dataset
print("Pre-loading models and clean test predictions...")
cache = {}
for name in manifest['dataset'].unique():
    with open(f"{MODELS_DIR}/{name}_rf.pkl", 'rb') as f:
        rf = pickle.load(f)
    with open(f"{META_DIR}/{name}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)

    clean_df = pd.read_csv(f"{PROCESSED_DIR}/{name}_test.csv")
    X_clean  = clean_df.drop(columns=['target']).values
    y_test   = clean_df['target'].values

    y_pred_clean = rf.predict(X_clean)
    n_classes    = len(meta['classes'])
    avg          = 'binary' if n_classes == 2 else 'macro'

    cache[name] = {
        'rf':            rf,
        'n_classes':     n_classes,
        'avg':           avg,
        'y_test':        y_test,
        'y_pred_clean':  y_pred_clean,
    }
    print(f"  Loaded: {name}")

print(f"\nRunning {len(manifest)} corruption evaluations...\n")

results = []
total   = len(manifest)

for i, exp in manifest.iterrows():
    dataset    = exp['dataset']
    tier       = exp['tier']
    corruption = exp['corruption_type']
    severity   = exp['severity']

    c         = cache[dataset]
    rf        = c['rf']
    n_classes = c['n_classes']
    avg       = c['avg']
    y_test    = c['y_test']
    y_pred_clean = c['y_pred_clean']

    base_acc = baseline.loc[dataset, 'accuracy']
    base_f1  = baseline.loc[dataset, 'f1']
    base_auc = baseline.loc[dataset, 'auc']

    # Load corrupted test set — fill NaN with 0 (StandardScaled mean)
    corr_df = pd.read_csv(exp['corrupted_file'])
    X_corr  = corr_df.drop(columns=['target']).fillna(0).values

    y_pred_corr = rf.predict(X_corr)
    y_prob_corr = rf.predict_proba(X_corr)

    acc_corr = round(accuracy_score(y_test, y_pred_corr), 4)
    f1_corr  = round(f1_score(y_test, y_pred_corr, average=avg, zero_division=0), 4)
    if n_classes == 2:
        auc_corr = round(roc_auc_score(y_test, y_prob_corr[:, 1]), 4)
    else:
        auc_corr = round(roc_auc_score(y_test, y_prob_corr,
                                        multi_class='ovr', average='macro'), 4)

    acc_drop = round(base_acc - acc_corr, 4)
    f1_drop  = round(base_f1  - f1_corr,  4)
    auc_drop = round(base_auc - auc_corr, 4)

    # Paired t-test: per-sample correctness on clean vs corrupted
    clean_correct = (y_pred_clean == y_test).astype(float)
    corr_correct  = (y_pred_corr  == y_test).astype(float)
    t_stat, p_value = stats.ttest_rel(clean_correct, corr_correct)
    significant = bool(p_value < P_VALUE_THRESHOLD)

    results.append({
        'dataset':         dataset,
        'tier':            tier,
        'corruption_type': corruption,
        'severity':        severity,
        'accuracy_base':   base_acc,
        'accuracy_corr':   acc_corr,
        'accuracy_drop':   acc_drop,
        'f1_base':         base_f1,
        'f1_corr':         f1_corr,
        'f1_drop':         f1_drop,
        'auc_base':        base_auc,
        'auc_corr':        auc_corr,
        'auc_drop':        auc_drop,
        't_stat':          round(float(t_stat), 4),
        'p_value':         round(float(p_value), 6),
        'significant':     significant,
    })

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{total}] {dataset} | {tier} | {corruption} | sev={severity} "
              f"| acc_drop={acc_drop:+.4f} | sig={significant}")

results_df = pd.DataFrame(results)
results_df.to_csv('step9_performance_results.csv', index=False)

print(f"\n{'='*60}")
print(f"{len(results_df)}/{total} experiments evaluated.")
print(f"Significant drops (p<0.05): {results_df['significant'].sum()}/{len(results_df)}")

print(f"\nMean accuracy drop by tier:")
print(results_df.groupby('tier')['accuracy_drop'].mean().round(4).to_string())

print(f"\nMean accuracy drop by corruption type:")
print(results_df.groupby('corruption_type')['accuracy_drop'].mean()
      .round(4).sort_values(ascending=False).to_string())

print(f"\nMean accuracy drop by severity:")
print(results_df.groupby('severity')['accuracy_drop'].mean().round(4).to_string())

