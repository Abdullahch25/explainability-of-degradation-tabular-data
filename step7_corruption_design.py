import pandas as pd
import numpy as np
import os
import pickle

SHAP_DIR = 'shap'
META_DIR = 'data/preprocessors'

SEVERITY_LEVELS  = [0.10, 0.30, 0.50]

# Corruption types and which feature types they apply to
CORRUPTION_TYPES = {
    'missing_mcar':    'both',
    'missing_mar':     'both',
    'missing_mnar':    'both',
    'gaussian_noise':  'numerical',
    'extreme_values':  'both',
    'value_swap':      'both',
    'scaling':         'numerical',
}

load_info = pd.read_csv('step2_load_summary.csv')
load_info = load_info[load_info['status'] == 'ok']

plan_rows  = []
tier_rows  = []   # one row per (dataset, tier) for inspection

for _, row in load_info.iterrows():
    name = row['name']
    print(f"--- {name} ---")

    imp_df = pd.read_csv(f"{SHAP_DIR}/{name}_shap_importance.csv")

    n_features = len(imp_df)
    tier_size  = max(1, n_features // 3)

    # Assign tiers by SHAP rank (rank 1 = most important)
    imp_df['tier'] = 'low'
    imp_df.loc[imp_df['rank'] <= tier_size, 'tier'] = 'high'
    imp_df.loc[
        (imp_df['rank'] > tier_size) & (imp_df['rank'] <= 2 * tier_size), 'tier'
    ] = 'medium'

    for tier in ['high', 'medium', 'low']:
        tier_feats = imp_df[imp_df['tier'] == tier]['feature'].tolist()

        has_num = any(f.startswith('num__') for f in tier_feats)
        has_cat = any(f.startswith('cat__') for f in tier_feats)

        tier_rows.append({
            'dataset':         name,
            'tier':            tier,
            'n_features':      len(tier_feats),
            'has_num':         has_num,
            'has_cat':         has_cat,
            'features':        '|'.join(tier_feats),
        })

        print(f"  {tier:6s}: {len(tier_feats)} features  "
              f"(num={has_num}, cat={has_cat})")

        for ctype, applies_to in CORRUPTION_TYPES.items():
            if applies_to == 'numerical' and not has_num:
                continue

            for severity in SEVERITY_LEVELS:
                plan_rows.append({
                    'dataset':         name,
                    'tier':            tier,
                    'n_features_tier': len(tier_feats),
                    'features':        '|'.join(tier_feats),
                    'has_num':         has_num,
                    'has_cat':         has_cat,
                    'corruption_type': ctype,
                    'severity':        severity,
                })

    print()

plan     = pd.DataFrame(plan_rows)
tier_map = pd.DataFrame(tier_rows)

plan.to_csv('step7_corruption_plan.csv', index=False)
tier_map.to_csv('step7_tier_map.csv', index=False)

print("=" * 60)
print(f"Total experiments  : {len(plan)}")
print(f"\nExperiments per corruption type:")
print(plan.groupby('corruption_type').size().rename('count').to_string())
print(f"\nExperiments per severity:")
print(plan.groupby('severity').size().rename('count').to_string())
print(f"\nExperiments per dataset:")
print(plan.groupby('dataset').size().rename('count').to_string())
