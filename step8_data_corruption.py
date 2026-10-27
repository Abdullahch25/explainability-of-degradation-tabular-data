import pandas as pd
import numpy as np
import os
import pickle

from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import GaussianNoise, Scaling

PROCESSED_DIR = 'data/processed'
CORRUPTED_DIR = 'data/corrupted'
META_DIR      = 'data/preprocessors'
os.makedirs(CORRUPTED_DIR, exist_ok=True)

plan     = pd.read_csv('step7_corruption_plan.csv')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Corruption helpers
def apply_missing(df, col, fraction, missingness):
    try:
        return MissingValues(column=col, fraction=fraction,
                             missingness=missingness).transform(df.copy())
    except Exception:
        # fallback to MCAR if MAR/MNAR fails (e.g. single-column tier)
        return MissingValues(column=col, fraction=fraction,
                             missingness='MCAR').transform(df.copy())

def apply_gaussian(df, col, fraction):
    return GaussianNoise(column=col, fraction=fraction).transform(df.copy())

def apply_scaling(df, col, fraction):
    return Scaling(column=col, fraction=fraction).transform(df.copy())

def apply_swap(df, col, fraction):
    return SwappedValues(column=col, fraction=fraction).transform(df.copy())

def apply_extreme(df, col, fraction):
    # Mark rows with MCAR, then replace NaN with extreme value
    corrupted = MissingValues(column=col, fraction=fraction,
                              missingness='MCAR').transform(df.copy())
    mask = corrupted[col].isna()
    if mask.any():
        if col.startswith('cat__'):
            corrupted.loc[mask, col] = 2.0
        else:
            mean = df[col].mean()
            std  = df[col].std()
            corrupted.loc[mask, col] = mean + 3 * std
    return corrupted

CORRUPT_FN = {
    'missing_mcar':   lambda df, col, frac: apply_missing(df, col, frac, 'MCAR'),
    'missing_mar':    lambda df, col, frac: apply_missing(df, col, frac, 'MAR'),
    'missing_mnar':   lambda df, col, frac: apply_missing(df, col, frac, 'MNAR'),
    'gaussian_noise': lambda df, col, frac: apply_gaussian(df, col, frac),
    'scaling':        lambda df, col, frac: apply_scaling(df, col, frac),
    'value_swap':     lambda df, col, frac: apply_swap(df, col, frac),
    'extreme_values': lambda df, col, frac: apply_extreme(df, col, frac),
}

# Main loop
manifest_rows = []
total = len(plan)

for i, exp in plan.iterrows():
    dataset        = exp['dataset']
    tier           = exp['tier']
    corruption     = exp['corruption_type']
    severity       = exp['severity']
    features       = exp['features'].split('|')
    sev_pct        = int(severity * 100)

    out_name = f"{dataset}_{tier}_{corruption}_{sev_pct}.csv"
    out_path = os.path.join(CORRUPTED_DIR, out_name)

    # Print progress every 50 experiments
    if i % 50 == 0:
        print(f"[{i+1}/{total}] {out_name}")

    # Load clean test set (reload fresh each experiment)
    test_df = pd.read_csv(f"{PROCESSED_DIR}/{dataset}_test.csv")

    # Apply corruption to each feature in the tier
    corrupted_df = test_df.copy()
    for col in features:
        if col not in corrupted_df.columns:
            continue
        fn = CORRUPT_FN[corruption]
        corrupted_df = fn(corrupted_df, col, severity)

    corrupted_df.to_csv(out_path, index=False)

    manifest_rows.append({
        'dataset':         dataset,
        'tier':            tier,
        'corruption_type': corruption,
        'severity':        severity,
        'n_features_corrupted': len(features),
        'corrupted_file':  out_path,
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv('step8_manifest.csv', index=False)

print(f"\n{'='*60}")
print(f"{len(manifest)}/{total} corrupted test sets saved.")
print(f"Corrupted data : {CORRUPTED_DIR}/")
print(f"Manifest       : step8_manifest.csv")
