import numpy as np
import pywt
from scipy.stats import entropy, kurtosis, skew
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
import os

# Paths
RAW_DATA_PATH = './data/X_eeg.npy'       # Shape: (samples, 32, timepoints)
LABEL_PATH = './data/y_labels.npy'
OUTPUT_X_PATH = './data/X_features.npy'
OUTPUT_Y_PATH = './data/y_labels.npy'

# EEG frequency bands (Hz) – for future use
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def extract_features(signal):
    """Extract features from a single EEG segment (shape: channels x timepoints)"""
    features = []

    for channel in signal:
        # Statistical features
        features.append(np.mean(channel))              # Mean
        features.append(np.std(channel))               # Std Dev
        features.append(entropy(np.abs(channel)))      # Entropy
        features.append(kurtosis(channel))             # Kurtosis
        features.append(skew(channel))                 # Skewness

        # Wavelet features (Discrete Wavelet Transform)
        coeffs = pywt.wavedec(channel, 'db4', level=3)
        for coeff in coeffs:
            features.append(np.mean(coeff))
            features.append(np.std(coeff))

    return features

def main():
    if not os.path.exists(RAW_DATA_PATH):
        print("❌ EEG data not found. Run load_data.py first.")
        return

    X = np.load(RAW_DATA_PATH)  # Shape: (samples, 32, timepoints)
    y = np.load(LABEL_PATH)     # Shape: (samples,)

    print(f"🔍 Extracting features from {X.shape[0]} EEG segments...")

    feature_matrix = []

    for segment in tqdm(X):
        features = extract_features(segment)
        feature_matrix.append(features)

    X_features = np.array(feature_matrix)
    print(f"✅ Feature extraction complete. Initial shape: {X_features.shape}")

    # Feature selection: Keep top 100 features
    print("🎯 Selecting top 100 informative features...")
    selector = SelectKBest(score_func=f_classif, k=100)
    X_selected = selector.fit_transform(X_features, y)

    # Save selected features
    np.save(OUTPUT_X_PATH, X_selected)
    np.save(OUTPUT_Y_PATH, y)

    print(f"✅ Selected shape: {X_selected.shape}")
    print(f"💾 Saved to: {OUTPUT_X_PATH}, {OUTPUT_Y_PATH}")

if __name__ == '__main__':
    main()
