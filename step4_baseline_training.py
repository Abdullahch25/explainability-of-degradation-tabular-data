import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = 'data/processed'
META_DIR      = 'data/preprocessors'
MODELS_DIR    = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

load_info = pd.read_csv('step2_load_summary.csv')
load_info = load_info[load_info['status'] == 'ok']

RANDOM_STATE = 42
N_ESTIMATORS = 100

summary_rows = []

for _, row in load_info.iterrows():
    name = row['name']
    print(f"--- {name} ---")

    train_df = pd.read_csv(f"{PROCESSED_DIR}/{name}_train.csv")
    X_train  = train_df.drop(columns=['target']).values
    y_train  = train_df['target'].values

    with open(f"{META_DIR}/{name}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)

    imbalanced   = meta['imbalanced']
    n_classes    = len(meta['classes'])
    class_weight = 'balanced' if imbalanced else None

    print(f"  Train size  : {len(X_train)}")
    print(f"  Classes     : {n_classes}, Imbalanced: {imbalanced}")
    print(f"  Class weight: {class_weight}")

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    model_path = f"{MODELS_DIR}/{name}_rf.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)

    print(f"  Saved       : {model_path}\n")

    summary_rows.append({
        'name':         name,
        'n_train':      len(X_train),
        'n_classes':    n_classes,
        'imbalanced':   imbalanced,
        'class_weight': str(class_weight),
        'model_path':   model_path,
        'status':       'ok',
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv('step4_baseline_summary.csv', index=False)

print("=" * 60)
print(f" {(summary['status'] == 'ok').sum()}/{len(summary)} models trained.")
print(f"Models saved : {MODELS_DIR}/")
print(summary[['name', 'n_train', 'n_classes', 'imbalanced', 'class_weight']].to_string())
