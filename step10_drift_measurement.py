import pandas as pd
import numpy as np
import warnings
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*ks_2samp.*')

try:
    from alibi_detect.cd import KSDrift
    HAS_ALIBI = True
    print("alibi-detect available — running KSDrift (primary) + SciPy KS (sanity check)")
except ImportError:
    HAS_ALIBI = False
    print("alibi-detect not installed — using SciPy KS test only")

PROCESSED_DIR = 'data/processed'
P_THRESHOLD   = 0.05

manifest = pd.read_csv('step8_manifest.csv')
plan_df  = pd.read_csv('step7_corruption_plan.csv')

# Build lookup: (dataset, tier) -> list of feature columns
tier_features = (
    plan_df[['dataset', 'tier', 'features']]
    .drop_duplicates(subset=['dataset', 'tier'])
    .assign(feature_list=lambda df: df['features'].str.split('|'))
    .set_index(['dataset', 'tier'])['feature_list']
    .to_dict()
)

# Pre-load clean test sets once per dataset
print("\nPre-loading clean test sets...")
clean_cache = {}
for name in manifest['dataset'].unique():
    df = pd.read_csv(f"{PROCESSED_DIR}/{name}_test.csv")
    clean_cache[name] = df
    print(f"  Loaded: {name}  ({len(df)} rows)")


alibi_detectors = {}
if HAS_ALIBI:
    print("\nBuilding KSDrift detectors per (dataset, tier)...")
    for (name, tier), feats in tier_features.items():
        clean_df  = clean_cache[name]
        avail     = [f for f in feats if f in clean_df.columns]
        if not avail:
            continue
        ref_data  = clean_df[avail].fillna(0).values.astype(np.float32)
        detector  = KSDrift(ref_data, p_val=P_THRESHOLD)
        alibi_detectors[(name, tier)] = (detector, avail)
    print(f"  Built {len(alibi_detectors)} detectors")

#Main loop
print(f"\nMeasuring drift for {len(manifest)} experiments...\n")
results = []
total   = len(manifest)

for i, exp in manifest.iterrows():
    dataset    = exp['dataset']
    tier       = exp['tier']
    corruption = exp['corruption_type']
    severity   = exp['severity']

    feats = tier_features.get((dataset, tier), [])
    clean_df = clean_cache[dataset]
    avail    = [f for f in feats if f in clean_df.columns]

    if not avail:
        continue

    corr_df  = pd.read_csv(exp['corrupted_file'])

    #SciPy KS test per feature (sanity check / primary if no alibi)
    ks_stats  = []
    ks_pvals  = []
    for col in avail:
        clean_col = clean_df[col].fillna(0).values
        corr_col  = corr_df[col].fillna(0).values if col in corr_df.columns else clean_col
        stat, pval = ks_2samp(clean_col, corr_col)
        ks_stats.append(stat)
        ks_pvals.append(pval)

    n_drifted_scipy = sum(p < P_THRESHOLD for p in ks_pvals)
    mean_ks_stat    = float(np.mean(ks_stats))
    max_ks_stat     = float(np.max(ks_stats))
    mean_ks_pval    = float(np.mean(ks_pvals))

    #alibi-detect KSDrift
    alibi_drift     = None
    alibi_pval      = None
    if HAS_ALIBI and (dataset, tier) in alibi_detectors:
        detector, det_feats = alibi_detectors[(dataset, tier)]
        test_data = corr_df[det_feats].fillna(0).values.astype(np.float32)
        try:
            pred          = detector.predict(test_data)
            alibi_drift   = bool(pred['data']['is_drift'])
            alibi_pval    = float(pred['data']['p_val'])
        except Exception:
            alibi_drift = None
            alibi_pval  = None

    results.append({
        'dataset':            dataset,
        'tier':               tier,
        'corruption_type':    corruption,
        'severity':           severity,
        'n_features_tested':  len(avail),
        'n_drifted_scipy':    n_drifted_scipy,
        'drift_pct_scipy':    round(n_drifted_scipy / len(avail) * 100, 1),
        'mean_ks_stat':       round(mean_ks_stat, 5),
        'max_ks_stat':        round(max_ks_stat, 5),
        'mean_ks_pval':       round(mean_ks_pval, 6),
        'alibi_drift':        alibi_drift,
        'alibi_pval':         round(alibi_pval, 6) if alibi_pval is not None else None,
    })

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{total}] {dataset} | {tier} | {corruption} | sev={severity} "
              f"| mean_ks={mean_ks_stat:.4f} | drifted={n_drifted_scipy}/{len(avail)}")

out_df = pd.DataFrame(results)
out_df.to_csv('step10_drift_results.csv', index=False)

print(f"\n{'='*60}")
print(f"Done. {len(out_df)}/{total} experiments measured.")
print(f"\nMean KS statistic by tier:")
print(out_df.groupby('tier')['mean_ks_stat'].mean().round(4).to_string())
print(f"\nMean % features drifted by tier:")
print(out_df.groupby('tier')['drift_pct_scipy'].mean().round(1).to_string())
print(f"\nMean KS statistic by corruption type:")
print(out_df.groupby('corruption_type')['mean_ks_stat']
      .mean().round(4).sort_values(ascending=False).to_string())
print(f"\nMean KS statistic by severity:")
print(out_df.groupby('severity')['mean_ks_stat'].mean().round(4).to_string())
if HAS_ALIBI:
    n_alibi = out_df['alibi_drift'].notna().sum()
    pct     = out_df['alibi_drift'].sum() / n_alibi * 100
    print(f"\nalibi-detect: drift detected in {out_df['alibi_drift'].sum()}/{n_alibi} "
          f"experiments ({pct:.1f}%)")

