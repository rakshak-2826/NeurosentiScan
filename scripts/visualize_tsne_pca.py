import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Paths ---
DATA_PATH = './data/datasets/eeg_data.csv'
OUTPUT_DIR = './outputs/visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and preprocess ---
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label'])
y = df['label'].str.strip()
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Label': y
})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Label', palette='Set2', alpha=0.7)
plt.title('PCA Visualization of EEG Feature Space')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_visualization.png'))
plt.close()

# --- t-SNE (on full X_scaled) ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'Label': y
})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Label', palette='Set2', alpha=0.7)
plt.title('t-SNE Visualization of EEG Feature Space')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualization.png'))
plt.close()

print("✅ PCA and t-SNE visualizations saved to:", OUTPUT_DIR)
