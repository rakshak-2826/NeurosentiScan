import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, LSTM, Bidirectional, TimeDistributed, Permute, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import gc
from joblib import dump

# ==== Paths ====
DATA_X = './data/X_features.npy'
DATA_Y = './data/y_labels.npy'
MODEL_DIR = './models'
OUTPUT_DIR = './outputs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Load and preprocess ====
X = np.load(DATA_X)  # shape (2132, 2548)
y = np.load(DATA_Y)

# Reshape to raw EEG format (samples, time, channels)
X = X.reshape((X.shape[0], 49, 52))  # pseudo-raw shape
scaler = StandardScaler()
X_2d = X.reshape((X.shape[0], -1))
X_scaled = scaler.fit_transform(X_2d).reshape(X.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.3, stratify=y_cat, random_state=42)

# ==== Utility ====
def save_confusion_and_preds(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_predictions.csv'), index=False)

# ==== Models ====
def build_eegnet():
    input_layer = Input(shape=(49, 52, 1))
    x = Conv2D(16, (1, 32), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

def build_bilstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(49, 52)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    return model

def build_cnn():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(49, 52)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    return model

# ==== Training wrapper ====
def train_model(name, model_builder, reshape_to=None):
    print(f"\n🚀 Training: {name}")
    model = model_builder()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early = EarlyStopping(patience=5, restore_best_weights=True)

    # Optional reshaping for 2D CNN input
    X_tr = X_train if reshape_to is None else X_train.reshape((-1, *reshape_to))
    X_te = X_test if reshape_to is None else X_test.reshape((-1, *reshape_to))

    hist = model.fit(X_tr, y_train, validation_split=0.2, epochs=50,
                     batch_size=32, callbacks=[early], verbose=1)

    loss, acc = model.evaluate(X_te, y_test)
    print(f"✅ {name} Accuracy: {acc:.4f}")

    # Save model
    model.save(os.path.join(MODEL_DIR, f'{name.lower()}_model.h5'))

    # Plot accuracy
    plt.figure()
    plt.plot(hist.history['accuracy'], label='Train')
    plt.plot(hist.history['val_accuracy'], label='Val')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name.lower()}_accuracy.png'))
    plt.close()

    # Predictions
    y_prob = model.predict(X_te)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    save_confusion_and_preds(name, y_true, y_pred)

    # Clear session
    del model
    K.clear_session()
    gc.collect()

# ==== Train all ====
train_model("EEGNet", build_eegnet, reshape_to=(49, 52, 1))
train_model("BiLSTM", build_bilstm)
train_model("CNN_1D", build_cnn)

# Save scaler and label encoder
dump(scaler, os.path.join(MODEL_DIR, 'raw_scaler.joblib'))
dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
