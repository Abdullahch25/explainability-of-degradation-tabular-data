import openml
import pandas as pd

suite = openml.study.get_suite('OpenML-CC18')
task_ids = suite.tasks

print(f"Total tasks in CC18: {len(task_ids)}")

rows = []
for task_id in task_ids:
    try:
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        qualities = dataset.qualities
        rows.append({
            'task_id': task_id,
            'dataset_id': task.dataset_id,
            'name': dataset.name,
            'n_instances': qualities.get('NumberOfInstances'),
            'n_features': qualities.get('NumberOfFeatures'),
            'missing_pct': qualities.get('PercentageOfMissingValues', 0),
            'n_classes': qualities.get('NumberOfClasses'),
        })
    except Exception as e:
        print(f"  Skipped task {task_id}: {e}")

df = pd.DataFrame(rows)

filtered = df[
    (df['n_instances'] >= 500) &
    (df['n_instances'] <= 100_000) &
    (df['n_features'] >= 5) &
    (df['n_features'] <= 100) &
    (df['missing_pct'] < 5) &
    (df['n_classes'] >= 2)
].copy()

filtered = filtered.sort_values('n_instances').reset_index(drop=True)

print(f"Datasets passing filters: {len(filtered)} / {len(df)}\n")
print(filtered[['name', 'dataset_id', 'task_id', 'n_instances', 'n_features', 'missing_pct', 'n_classes']].to_string())

filtered.to_csv('step1_filtered_datasets.csv', index=False)

# Size buckets to guide stratified selection
small  = filtered[filtered['n_instances'] <  2_000]
medium = filtered[(filtered['n_instances'] >= 2_000) & (filtered['n_instances'] < 20_000)]
large  = filtered[filtered['n_instances'] >= 20_000]

print(f"\nSize breakdown:")
print(f"  Small  (<2k rows):      {len(small)}")
print(f"  Medium (2k-20k rows):   {len(medium)}")
print(f"  Large  (>20k rows):     {len(large)}")

# Final selection of 13 datasets
SELECTED_NAMES = [
    'cmc',
    'diabetes',
    'vehicle',
    'credit-g',
    'qsar-biodeg',
    'ozone-level-8hr',
    'segment',
    'spambase',
    'phoneme',
    'satimage',
    'bank-marketing',
    'electricity',
    'connect-4',
]

selected = filtered[filtered['name'].isin(SELECTED_NAMES)].copy()
selected = selected.sort_values('n_instances').reset_index(drop=True)

def size_label(n):
    if n < 2000:
        return 'small'
    elif n < 20000:
        return 'medium'
    else:
        return 'large'

selected['size'] = selected['n_instances'].apply(size_label)

print(f"\n{'='*70}")
print(f"Final selection: {len(selected)} datasets")
print(f"{'='*70}")
print(selected[['name', 'dataset_id', 'task_id', 'n_instances', 'n_features', 'missing_pct', 'n_classes', 'size']].to_string())

selected.to_csv('step1_selected_datasets.csv', index=False)
