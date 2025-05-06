import numpy as np
from tqdm import tqdm
import os

# Paths
RAW_DATA_PATH = './data/X_eeg.npy'
LABEL_PATH = './data/y_labels.npy'
OUTPUT_PATH = './data/X_dl.npy'
NORMALIZED = True
DOWNSAMPLE_FACTOR = 2  # Set to 1 for no downsampling

def z_score_normalize(data):
    """Per-channel z-score normalization"""
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)

def downsample(data, factor):
    """Downsamples along the time axis"""
    return data[:, :, ::factor]

def preprocess_eeg(X, normalize=True, downsample_factor=1):
    print(f"🔧 Preprocessing {X.shape[0]} EEG samples...")
    processed = []

    for segment in tqdm(X):
        if normalize:
            segment = z_score_normalize(segment)
        if downsample_factor > 1:
            segment = segment[:, ::downsample_factor]
        processed.append(segment)

    return np.array(processed)

def main():
    # Load
    if not os.path.exists(RAW_DATA_PATH):
        print("❌ Raw EEG file not found. Run load_sam40.py first.")
        return

    X = np.load(RAW_DATA_PATH)
    y = np.load(LABEL_PATH)

    # Process
    X_processed = preprocess_eeg(X, normalize=NORMALIZED, downsample_factor=DOWNSAMPLE_FACTOR)

    # Save
    np.save(OUTPUT_PATH, X_processed)
    print(f"✅ Preprocessing complete.")
    print(f"📐 Final shape: {X_processed.shape}")
    print(f"💾 Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
