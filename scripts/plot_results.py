import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
OUTPUT_DIR = './outputs/'
GRAPH_DIR = './graph/'
os.makedirs(GRAPH_DIR, exist_ok=True)

# Models
ml_models = ['SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression', 'DecisionTree']
dl_models = ['CNN_1D', 'LSTM', 'CNN_LSTM']

def load_reports_from_models(model_list):
    reports = {}
    for model in model_list:
        path = os.path.join(OUTPUT_DIR, f'{model}_report.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0)
            if 'accuracy' in df.index:
                acc = df.loc['accuracy']['f1-score'] if 'f1-score' in df.columns else df.loc['accuracy']['precision']
                reports[model] = acc
        except Exception as e:
            print(f"❌ Error loading {model}_report.csv: {e}")
    return pd.DataFrame(list(reports.items()), columns=['Model', 'Accuracy'])

# Load results
ml_df = load_reports_from_models(ml_models)
dl_df = load_reports_from_models(dl_models)

# 📊 Plot ML Accuracies
if not ml_df.empty:
    ml_df = ml_df.sort_values(by='Accuracy', ascending=False)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='Model', y='Accuracy', data=ml_df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f')
    plt.title("ML Models - Accuracy Comparison")
    plt.ylim(ml_df['Accuracy'].min() - 0.02, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'ml_model_accuracy_comparison_eval.png'))
    plt.close()

# 📊 Plot DL Accuracies
if not dl_df.empty:
    dl_df = dl_df.sort_values(by='Accuracy', ascending=False)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='Model', y='Accuracy', data=dl_df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f')
    plt.title("DL Models - Accuracy Comparison")
    plt.ylim(dl_df['Accuracy'].min() - 0.02, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'dl_model_accuracy_comparison_eval.png'))
    plt.close()

# ✅ Combined Accuracy Table
combined_df = pd.concat([ml_df, dl_df], axis=0).sort_values(by='Accuracy', ascending=False)
print("\n=== Model Accuracy Comparison ===")
print(combined_df)
