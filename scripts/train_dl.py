import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc
import pandas as pd
from joblib import dump

# Paths
X_PATH = './data/X_features.npy'
Y_PATH = './data/y_labels.npy'
MODEL_DIR = './models/'
OUTPUT_DIR = './outputs/'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess
X = np.load(X_PATH)
y = np.load(Y_PATH)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for deep learning
X = X.reshape(X.shape[0], 49, 52)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

def build_and_train_model(name, build_fn):
    print(f"\n🚀 Training: {name}")
    model = build_fn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ {name} Accuracy: {acc:.4f}")
    model.save(os.path.join(MODEL_DIR, f'{name.lower()}_model.h5'))

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name.lower()}_accuracy.png'))
    plt.close()

    # Confusion matrix + Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name.lower()}_confusion_matrix.png'))
    plt.close()

    # Save predictions for statistical testing
    df_preds = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_classes
    })
    df_preds.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions.csv'), index=False)

    # Clean up
    del model
    K.clear_session()
    gc.collect()

# Model builders
def build_cnn():
    return Sequential([
        Conv1D(128, 3, activation='relu', input_shape=(49, 52)),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])

def build_lstm():
    return Sequential([
        LSTM(64, input_shape=(49, 52)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

def build_cnn_lstm():
    return Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(49, 52)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

# Train models
build_and_train_model("CNN", build_cnn)
build_and_train_model("LSTM", build_lstm)
build_and_train_model("CNN_LSTM", build_cnn_lstm)

# ✅ Save the label encoder and scaler for evaluation reuse
dump(scaler, os.path.join(MODEL_DIR, 'dl_scaler.joblib'))
dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
