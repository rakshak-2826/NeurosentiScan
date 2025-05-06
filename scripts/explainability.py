import os
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

# Prevent memory overload from OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Paths
DATA_PATH = './data/datasets/eeg_data.csv'
MODEL_DIR = './models'
EXPLAIN_DIR = './explainability/'
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# Load original dataset
df = pd.read_csv(DATA_PATH)
X_raw = df.drop(columns=['label'])
y_raw = df['label']
class_names = sorted(y_raw.unique())
feature_names = X_raw.columns.tolist()

# How many samples per model
N_SAMPLES = 3

# Target models
models = ['SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression', 'DecisionTree']

for model_name in models:
    raw_model_path = os.path.join(MODEL_DIR, f"{model_name}_raw.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.joblib")

    if not os.path.exists(raw_model_path) or not os.path.exists(scaler_path):
        print(f"❌ Skipping {model_name} — model or scaler not found")
        continue

    print(f"\n🔍 Generating LIME explanations for: {model_name}")

    try:
        # Load model and scaler
        model = joblib.load(raw_model_path)
        scaler = joblib.load(scaler_path)

        # Scale full-feature data
        X_scaled = scaler.transform(X_raw)

        # Subset for explanation
        X_subset = X_scaled[:N_SAMPLES]
        X_original = X_raw.iloc[:N_SAMPLES]

        # Build LIME explainer
        explainer = LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        # Explain each sample
        for i in range(N_SAMPLES):
            exp = explainer.explain_instance(
                data_row=X_subset[i],
                predict_fn=model.predict_proba,
                num_features=10
            )
            html_path = os.path.join(EXPLAIN_DIR, f"lime_{model_name.lower()}_sample_{i+1}.html")
            exp.save_to_file(html_path)

        print(f"✅ {model_name}: Saved {N_SAMPLES} explanations to {EXPLAIN_DIR}")

    except Exception as e:
        print(f"❌ Failed to explain {model_name}: {str(e)}")

print("\n🎉 All LIME explanations completed.")
