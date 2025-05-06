import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter

# Paths
OUTPUT_DIR = './outputs'
PREDICTION_FILES = [
    'SVM_predictions.csv',
    'RandomForest_predictions.csv',
    'KNN_predictions.csv',
    'LogisticRegression_predictions.csv',
    'NaiveBayes_predictions.csv',
    'DecisionTree_predictions.csv',
    'CNN_predictions.csv',
    'LSTM_predictions.csv',
    'CNN_LSTM_predictions.csv',
    'CNN_1D_predictions.csv',
    'EEGNet_predictions.csv',
    'BiLSTM_predictions.csv'
]

print("\n🔍 Loading prediction files for ensembling...")
all_preds = []
valid_files = []

# Load available predictions
for file in PREDICTION_FILES:
    path = os.path.join(OUTPUT_DIR, file)
    if not os.path.exists(path):
        print(f"❌ Missing: {file}")
        continue
    df = pd.read_csv(path)
    all_preds.append(df)
    valid_files.append(file)
    print(f"✅ Loaded: {file} ({len(df)} samples)")

# Ensure same number of samples
min_len = min(len(df) for df in all_preds)
all_preds = [df.iloc[:min_len] for df in all_preds]

# Stack predictions
y_true = all_preds[0]['y_true'].values
all_model_preds = np.stack([df['y_pred'].values for df in all_preds], axis=1)

# Majority voting
y_ensemble = []
for row in all_model_preds:
    vote = Counter(row).most_common(1)[0][0]
    y_ensemble.append(vote)
y_ensemble = np.array(y_ensemble)

# Accuracy
ensemble_acc = accuracy_score(y_true, y_ensemble)
print(f"\n📊 Ensemble Accuracy (from {len(valid_files)} models): {ensemble_acc:.4f}")

# Save ensemble results
ensemble_df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_ensemble
})
output_path = os.path.join(OUTPUT_DIR, 'Ensemble_predictions.csv')
ensemble_df.to_csv(output_path, index=False)
print(f"💾 Saved to: {output_path}")
