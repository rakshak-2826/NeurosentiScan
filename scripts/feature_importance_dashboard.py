# scripts/feature_importance_dashboard.py

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
DATA_PATH = './data/datasets/eeg_data.csv'
MODEL_DIR = './models/'
OUTPUT_PATH = './outputs/feature_importance_dashboard.png'
os.makedirs('./outputs', exist_ok=True)

# Load feature names
df = pd.read_csv(DATA_PATH)
feature_names = df.drop(columns=['label']).columns

# ML models to include
ML_MODELS = ['SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression', 'DecisionTree']
importance_df = pd.DataFrame(index=feature_names)

for model_name in ML_MODELS:
    path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        print(f"❌ Skipping {model_name} — model not found.")
        continue

    model = joblib.load(path)

    # Access final trained estimator
    if hasattr(model, 'best_estimator_'):
        pipeline = model.best_estimator_
    else:
        print(f"❌ Skipping {model_name} — no best_estimator_ found.")
        continue

    clf = pipeline.named_steps['clf']

    try:
        if hasattr(clf, 'coef_'):
            coef = clf.coef_
            if coef.ndim == 2 and coef.shape[0] > 1:
                importance = abs(coef).mean(axis=0)  # Average across classes
            else:
                importance = abs(coef).flatten()
        elif hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        else:
            print(f"⚠️ {model_name} does not support feature importance.")
            continue

        if len(importance) != len(feature_names):
            print(f"❌ Skipping {model_name} — importance shape mismatch ({len(importance)} vs {len(feature_names)})")
            continue

        importance_df[model_name] = importance

    except Exception as e:
        print(f"❌ Failed to extract from {model_name}: {e}")

# Drop any empty columns
importance_df = importance_df.dropna(axis=1, how='all')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(importance_df.T, cmap='viridis', cbar_kws={'label': 'Feature Importance'})
plt.xlabel("Features")
plt.ylabel("Models")
plt.title("EEG Feature Importance Across Models")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.close()

print(f"✅ Feature importance heatmap saved to: {OUTPUT_PATH}")
