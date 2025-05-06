import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from joblib import load

# Paths
DATA_X = './data/X_features.npy'
DATA_Y = './data/y_labels.npy'
MODEL_DIR = './models/'
OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
X = np.load(DATA_X)
y = np.load(DATA_Y)

# Load saved label encoder and DL scaler
le = load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
scaler = load(os.path.join(MODEL_DIR, 'dl_scaler.joblib'))

# Encode labels
y_encoded = le.transform(y)

# --- ML MODEL EVALUATION ---
print("📊 Evaluating ML Models")
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

def save_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

def save_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_report.csv"))
    return report['accuracy']

def save_predictions(y_true, y_pred, model_name):
    df_preds = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_preds.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_predictions.csv'), index=False)

ml_models = ['SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression', 'DecisionTree']
for model_name in ml_models:
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        continue
    print(f"🔍 Evaluating {model_name}")
    model = load(model_path)
    y_pred = model.predict(X_test_ml)
    acc = save_report(y_test_ml, y_pred, model_name)
    save_conf_matrix(y_test_ml, y_pred, model_name)
    save_predictions(y_test_ml, y_pred, model_name)
    print(f"✅ {model_name} Accuracy: {acc:.4f}")

# --- DL MODEL EVALUATION ---
print("\n📊 Evaluating DL Models")

# Apply scaler and reshape
X_scaled = scaler.transform(X)
X_dl = X_scaled.reshape((X_scaled.shape[0], 49, 52))
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
y_test_cat = to_categorical(y_test_dl, num_classes=3)

# Manually mapped model names to filenames
dl_model_map = {
    'CNN_1D': 'cnn_model.h5',
    'LSTM': 'lstm_model.h5',
    'CNN_LSTM': 'cnn_lstm_model.h5'
}

for model_name, file_name in dl_model_map.items():
    model_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(model_path):
        continue
    print(f"🔍 Evaluating {model_name}")
    model = load_model(model_path)
    y_pred_probs = model.predict(X_test_dl)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    acc = save_report(y_test_dl, y_pred_classes, model_name)
    save_conf_matrix(y_test_dl, y_pred_classes, model_name)
    save_predictions(y_test_dl, y_pred_classes, model_name)
    print(f"✅ {model_name} Accuracy: {acc:.4f}")
