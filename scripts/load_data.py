import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

DATA_DIR = './data/filtered_data/'
OUTPUT_X = './data/X_eeg.npy'
OUTPUT_Y = './data/y_labels.npy'

eeg_list = []
label_list = []

mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
print(f"🔍 Found {len(mat_files)} .mat files. Starting to load...")

def find_eeg_key(mat):
    for k in mat:
        if not k.startswith("__"):
            val = mat[k]
            if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] in [32, 64]:
                return k
    return None

for i, file in enumerate(tqdm(mat_files)):
    mat = loadmat(os.path.join(DATA_DIR, file))
    key = find_eeg_key(mat)
    if key is None:
        print(f"⚠️ Skipping {file} — no valid EEG data found.")
        continue

    data = mat[key]
    eeg_list.append(data)

    # First 240 = relaxed, Next 240 = stress
    label = 0 if i < 240 else 1
    label_list.append(label)

X = np.array(eeg_list)
y = np.array(label_list)

print(f"\n✅ Done. Loaded {X.shape[0]} EEG segments.")
print(f"📐 EEG shape: {X.shape}, Labels shape: {y.shape} | Class counts: {np.bincount(y)}")

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
print(f"💾 Saved: {OUTPUT_X} and {OUTPUT_Y}")
