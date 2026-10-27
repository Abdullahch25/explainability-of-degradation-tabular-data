import openml
import pandas as pd
import os

RAW_DIR = 'data/raw'
os.makedirs(RAW_DIR, exist_ok=True)

selected = pd.read_csv('step1_selected_datasets.csv')
print(f"Loading {len(selected)} datasets...\n")

summary_rows = []

for _, row in selected.iterrows():
    name        = row['name']
    dataset_id  = int(row['dataset_id'])
    task_id     = int(row['task_id'])

    print(f"--- {name} (dataset_id={dataset_id}) ---")
    try:
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )

        X, y, categorical_mask, feature_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute,
        )

        df = X.copy()
        df['target'] = y

        n_rows, n_cols = df.shape
        n_missing     = df.isnull().sum().sum()
        missing_pct   = round(100 * n_missing / (n_rows * n_cols), 3)
        n_classes     = df['target'].nunique()
        class_dist    = df['target'].value_counts(normalize=True).round(3).to_dict()
        cat_cols      = [f for f, is_cat in zip(feature_names, categorical_mask) if is_cat]

        print(f"  Shape      : {n_rows} rows x {n_cols - 1} features (+target)")
        print(f"  Target     : '{dataset.default_target_attribute}' — {n_classes} classes")
        print(f"  Missing    : {n_missing} values ({missing_pct}%)")
        print(f"  Cat cols   : {len(cat_cols)} {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")
        print(f"  Class dist : {class_dist}\n")

        out_path = os.path.join(RAW_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)

        summary_rows.append({
            'name':           name,
            'dataset_id':     dataset_id,
            'task_id':        task_id,
            'n_rows':         n_rows,
            'n_features':     n_cols - 1,
            'n_missing':      n_missing,
            'missing_pct':    missing_pct,
            'n_classes':      n_classes,
            'n_cat_features': len(cat_cols),
            'target_col':     dataset.default_target_attribute,
            'saved_to':       out_path,
            'status':         'ok',
        })

    except Exception as e:
        print(f"  ERROR: {e}\n")
        summary_rows.append({
            'name': name, 'dataset_id': dataset_id, 'task_id': task_id,
            'status': f'error: {e}',
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv('step2_load_summary.csv', index=False)

print("=" * 60)
print(f" {(summary['status'] == 'ok').sum()}/{len(summary)} loaded successfully.")
print(f"Raw CSVs saved to: {RAW_DIR}/")

print("\n", summary[['name', 'n_rows', 'n_features', 'n_missing', 'n_classes', 'status']].to_string())
