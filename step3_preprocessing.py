import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROCESSED_DIR  = 'data/processed'
META_DIR       = 'data/preprocessors'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

selected  = pd.read_csv('step1_selected_datasets.csv')
load_info = pd.read_csv('step2_load_summary.csv')
load_info = load_info[load_info['status'] == 'ok']

RANDOM_STATE = 42
TEST_SIZE    = 0.2

summary_rows = []

for _, row in load_info.iterrows():
    name       = row['name']
    target_col = row['target_col']
    print(f"--- {name} ---")

    df = pd.read_csv(f"data/raw/{name}.csv")

    # step2 saved the target under the column name 'target'
    X = df.drop(columns=['target'])
    y = df['target']

    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  Features   : {len(num_cols)} numerical, {len(cat_cols)} categorical")

    # Encode target to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    classes = le.classes_.tolist()

    # Check class imbalance (minority class %)
    class_counts  = pd.Series(y_enc).value_counts(normalize=True)
    minority_pct  = round(class_counts.min() * 100, 1)
    imbalanced    = minority_pct < 20.0
    print(f"  Target     : {len(classes)} classes {classes}")
    print(f"  Minority   : {minority_pct}% {'<-- IMBALANCED' if imbalanced else ''}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    # Build preprocessing pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Get expanded feature names (OHE creates new columns per category)
    feature_names_out = preprocessor.get_feature_names_out().tolist()
    n_features_out = len(feature_names_out)
    print(f"  OHE expanded: {len(num_cols) + len(cat_cols)} -> {n_features_out} features")

    train_df = pd.DataFrame(X_train_proc, columns=feature_names_out)
    train_df['target'] = y_train.values if hasattr(y_train, 'values') else y_train

    test_df = pd.DataFrame(X_test_proc, columns=feature_names_out)
    test_df['target'] = y_test.values if hasattr(y_test, 'values') else y_test

    train_df.to_csv(f"{PROCESSED_DIR}/{name}_train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/{name}_test.csv",  index=False)

    # Save metadata (preprocessor + column info) for downstream steps
    meta = {
        'name':               name,
        'preprocessor':       preprocessor,
        'label_encoder':      le,
        'num_cols':           num_cols,
        'cat_cols':           cat_cols,
        'feature_names':      feature_names_out,
        'target_col':         target_col,
        'classes':            classes,
        'n_train':            len(train_df),
        'n_test':             len(test_df),
        'imbalanced':         imbalanced,
        'minority_pct':       minority_pct,
    }
    with open(f"{META_DIR}/{name}_meta.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"  Split      : {len(train_df)} train / {len(test_df)} test")
    print(f"  Saved      : {PROCESSED_DIR}/{name}_train.csv & _test.csv\n")

    summary_rows.append({
        'name':           name,
        'n_train':        len(train_df),
        'n_test':         len(test_df),
        'n_num_cols':     len(num_cols),
        'n_cat_cols':     len(cat_cols),
        'n_features_out': n_features_out,
        'n_classes':      len(classes),
        'minority_pct':   minority_pct,
        'imbalanced':     imbalanced,
        'status':         'ok',
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv('step3_preprocessing_summary.csv', index=False)

print("=" * 60)
print(f" {(summary['status'] == 'ok').sum()}/{len(summary)} preprocessed successfully.")
print(f"Processed splits : {PROCESSED_DIR}/")
print(f"Preprocessors    : {META_DIR}/")
print(summary[['name', 'n_train', 'n_test', 'n_num_cols', 'n_cat_cols', 'n_features_out', 'minority_pct', 'imbalanced']].to_string())
