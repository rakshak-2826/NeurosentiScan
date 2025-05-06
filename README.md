# 🧠 NeuroSentiScan - EEG-based Emotion Classification

**NeuroSentiScan** is a comprehensive research-oriented offline system designed to classify emotional states—**Negative**, **Neutral**, and **Positive**—from EEG signals. It implements a robust pipeline that includes preprocessing, feature extraction, model training, comparative evaluation, visual analytics, and interpretability using both classical ML models and advanced Deep Learning architectures.

The system is tailored for academic and applied neuroscience research, particularly in mental health monitoring, emotion recognition, and neurofeedback applications.

---

## 📦 Project Overview

- **Dataset**:  
  A feature-engineered EEG dataset consisting of **2132 samples** with **2548 extracted features** derived from raw EEG signals. Each sample corresponds to a labeled emotional state (NEGATIVE, NEUTRAL, POSITIVE).

- **Problem Type**:  
  Multi-class classification task focused on mapping EEG feature vectors to one of three emotional states.

- **EEG Feature Types**:
  - Time-domain: mean, variance, min, max, skewness, kurtosis
  - Frequency-domain: FFT, power spectral density
  - Statistical and correlation metrics

- **Classes**:  
  - `0`: NEGATIVE  
  - `1`: NEUTRAL  
  - `2`: POSITIVE  

- **Machine Learning Models**:
  - ✅ Support Vector Machine (SVM)
  - ✅ Random Forest
  - ✅ K-Nearest Neighbors (KNN)
  - ✅ Logistic Regression
  - ✅ Naive Bayes
  - ✅ Decision Tree

- **Deep Learning Models**:
  - ✅ 1D Convolutional Neural Network (CNN)
  - ✅ LSTM (Long Short-Term Memory)
  - ✅ CNN + LSTM Hybrid
  - ✅ EEGNet — A compact CNN for EEG decoding
  - ✅ BiLSTM with optional attention mechanism

- **Data Input for DL**:
  - Reshaped to `(49, 52)` for CNN/LSTM compatibility
  - Normalized using `StandardScaler`
  - Labels encoded using `LabelEncoder` and one-hot encoded for categorical training

---

## 📊 Evaluation Strategy

The evaluation pipeline includes comprehensive metrics and techniques to benchmark model performance and ensure fairness across all approaches.

### ✅ Core Evaluation Metrics:
- **Accuracy**: Correct predictions / total predictions
- **Precision**: Class-specific prediction accuracy
- **Recall**: Sensitivity per class (True Positive Rate)
- **F1-Score**: Harmonic mean of precision and recall

### 📈 Evaluation Output Includes:
- Confusion matrix per model (as `.png`)
- Classification report per model (`.csv`)
- Accuracy-vs-Epoch plots for DL models
- Bar plots comparing ML vs DL vs Ensemble models

### 🧠 Advanced Evaluation:
- **Model Ensembling**:
  - Uses majority voting across 12 models
  - Yields improved generalization and accuracy

- **Statistical Tests**:
  - **Paired t-test**: Compares model prediction differences
  - **McNemar's test**: Analyzes pairwise label prediction agreement
  - Results saved in: `statistical_tests_results.csv`

- **Interpretability (LIME)**:
  - Explains model decisions on a per-sample basis
  - Useful for local decision inspection (model reasoning)

- **Optional SHAP/Correlation Analysis**:
  - Feature-level impact scores (planned extensions)

---

## 🔄 Workflow Summary

1. **Data Preprocessing**
   - Encode labels with `LabelEncoder`
   - Normalize features using `StandardScaler`
   - Reshape for DL: `(samples, 49, 52)`

2. **ML Training**
   - Models trained using `GridSearchCV`
   - Saves `.joblib` model files + predictions

3. **DL Training**
   - CNN, LSTM, CNN+LSTM, BiLSTM, EEGNet
   - Trained with EarlyStopping + Dropout
   - Models saved as `.h5` with graphs

4. **Prediction & Ensemble**
   - `generate_predictions.py`: Saves aligned `*_predictions.csv`
   - `ensemble_predictions.py`: Majority voting ensembling

5. **Statistical Testing**
   - `stats_tests.py`: Paired t-test + McNemar test

6. **Visual Analytics**
   - PCA/t-SNE for class separation
   - Accuracy bar charts for all model types

7. **Model Explainability**
   - `explainability.py`: Generates LIME explanation plots

---

## 🧪 Final Output

- 📊 Confusion matrix (per model)
- 📄 Classification report (`.csv`)
- 📈 Accuracy plots (training and comparison)
- 📂 Saved model files: `.joblib` (ML), `.h5` (DL)
- 🧠 Ensemble predictions: `Ensemble_predictions.csv`
- 📉 PCA / t-SNE visualizations
- 📄 Statistical results: `statistical_tests_results.csv`
- 🔍 LIME-based interpretability plots

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧰 Scripts

```bash
# ML Training & Evaluation
python scripts/train_ml.py

# DL Training (CNN, LSTM, CNN+LSTM)
python scripts/train_dl.py

# DL on Raw EEG (EEGNet, BiLSTM, CNN_1D)
python scripts/train_dl_raw.py

# Unified prediction generation
python scripts/generate_predictions.py

# Ensemble voting of model outputs
python scripts/ensemble_predictions.py

# Statistical tests (paired t-test & McNemar)
python scripts/stats_tests.py

# Interpretability with LIME
python scripts/explainability.py

# Dimensionality Reduction Visuals
python scripts/tsne_pca.py

# Accuracy Bar Plots
python scripts/plot_results.py

# Full Pipeline
python main.py
```

---

## 📄 License

For academic and non-commercial use only.  
© NeuroSentiScan – 2025