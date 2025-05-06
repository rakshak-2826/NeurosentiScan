# scripts/generate_predictions.py

import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = './data/datasets/eeg_data.csv'
MODEL_DIR = './models'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label'])
y = df['label'].str.strip()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# =================== ML MODELS ===================
ml_models = [
    'SVM', 'RandomForest', 'KNN', 'NaiveBayes',
    'LogisticRegression', 'DecisionTree'
]

for name in ml_models:
    model_path = os.path.join(MODEL_DIR, f'{name}.joblib')
    if not os.path.exists(model_path):
        print(f"❌ Skipping {name} — model not found.")
        continue

    model = load(model_path)
    y_pred = model.predict(X_test)

    df_pred = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_pred.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions.csv'), index=False)
    print(f"✅ Saved ML predictions for {name}")

# =================== DL MODELS ===================
dl_models = {
    'CNN': 'cnn_model.h5',
    'LSTM': 'lstm_model.h5',
    'CNN_LSTM': 'cnn_lstm_model.h5',
    'CNN_1D': 'cnn_1d_model.h5',
    'EEGNet': 'eegnet_model.h5',
    'BiLSTM': 'bilstm_model.h5'
}

scaler_path = os.path.join(MODEL_DIR, 'dl_scaler.joblib')
if not os.path.exists(scaler_path):
    print("❌ Missing DL scaler. Skipping DL predictions.")
else:
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 49, 52))

    _, X_test_dl, _, y_test_dl = train_test_split(
        X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

    for name, fname in dl_models.items():
        model_path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(model_path):
            print(f"❌ Skipping {name} — model not found.")
            continue

        model = load_model(model_path)
        y_prob = model.predict(X_test_dl)
        y_pred = np.argmax(y_prob, axis=1)

        df_pred = pd.DataFrame({
            'y_true': y_test_dl,
            'y_pred': y_pred
        })
        df_pred.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions.csv'), index=False)
        print(f"✅ Saved DL predictions for {name}")
