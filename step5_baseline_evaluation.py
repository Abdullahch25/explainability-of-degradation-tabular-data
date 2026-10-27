import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROCESSED_DIR = 'data/processed'
META_DIR      = 'data/preprocessors'
MODELS_DIR    = 'models'

load_info = pd.read_csv('step2_load_summary.csv')
load_info = load_info[load_info['status'] == 'ok']

summary_rows = []

for _, row in load_info.iterrows():
    name = row['name']
    print(f"--- {name} ---")

    test_df = pd.read_csv(f"{PROCESSED_DIR}/{name}_test.csv")
    X_test  = test_df.drop(columns=['target']).values
    y_test  = test_df['target'].values

    with open(f"{META_DIR}/{name}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)

    with open(f"{MODELS_DIR}/{name}_rf.pkl", 'rb') as f:
        rf = pickle.load(f)

    n_classes = len(meta['classes'])

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)

    avg = 'binary' if n_classes == 2 else 'macro'
    f1  = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

    if n_classes == 2:
        auc = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
    else:
        auc = round(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'), 4)

    print(f"  Accuracy : {accuracy}")
    print(f"  F1       : {f1}  (averaging: {avg})")
    print(f"  AUC      : {auc}\n")

    summary_rows.append({
        'name':      name,
        'n_test':    len(X_test),
        'n_classes': n_classes,
        'accuracy':  accuracy,
        'f1':        f1,
        'auc':       auc,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv('step5_baseline_evaluation.csv', index=False)

print("=" * 60)
print(summary[['name', 'accuracy', 'f1', 'auc']].to_string())
