import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from tqdm import tqdm

# Paths
DATA_PATH = './data/datasets/eeg_data.csv'
MODEL_DIR = './models'
OUTPUT_DIR = './outputs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['label'])
    y = df['label'].str.strip()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

def plot_conf_matrix(y_true, y_pred, model_name, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def log_classification_report(y_true, y_pred, model_name, label_encoder):
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_report.csv'))
    return report['accuracy']

def train_model(name, model, param_grid, X_train, X_test, y_train, y_test, label_encoder):
    print(f"\n🚀 Training: {name}")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    clf = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Save performance metrics
    acc = log_classification_report(y_test, y_pred, name, label_encoder)
    plot_conf_matrix(y_test, y_pred, name, label_encoder.classes_)

    # Save full pipeline
    dump(clf, os.path.join(MODEL_DIR, f'{name}.joblib'))

    # Save raw classifier and scaler separately for LIME/SHAP
    best_pipeline = clf.best_estimator_
    dump(best_pipeline.named_steps['clf'], os.path.join(MODEL_DIR, f'{name}_raw.joblib'))
    dump(best_pipeline.named_steps['scaler'], os.path.join(MODEL_DIR, f'{name}_scaler.joblib'))

    # Save predictions
    df_preds = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_preds.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions.csv'), index=False)

    print(f"✅ {name}: Accuracy = {acc:.4f} | Model, metrics, raw components saved.")
    return acc, y_test, y_pred

def main():
    print("🔍 Loading EEG data from CSV...")
    X, y_encoded, label_encoder = load_data()
    print(f"📊 Data shape: {X.shape}, Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

    models = {
        'SVM': {
            'model': SVC(probability=True),
            'params': {'clf__C': [1], 'clf__kernel': ['rbf']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {'clf__n_estimators': [100], 'clf__max_depth': [None]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'clf__n_neighbors': [5]}
        },
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {}
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'clf__C': [1.0]}
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(),
            'params': {'clf__max_depth': [None, 10, 20]}
        }
    }

    results = {}
    all_preds = {}

    for name, cfg in tqdm(models.items()):
        acc, y_true, y_pred = train_model(name, cfg['model'], cfg['params'], X_train, X_test, y_train, y_test, label_encoder)
        results[name] = acc
        all_preds[name] = {'y_true': y_true, 'y_pred': y_pred}

    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison (Multiclass)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_accuracy_comparison_multiclass.png"))
    print("📊 Comparison plot saved.")

if __name__ == '__main__':
    main()
