import os
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix

# Paths
OUTPUT_DIR = './outputs/'
MODEL_PAIRS = [
    ('SVM', 'RandomForest'),
    ('CNN_LSTM', 'LSTM'),
    ('CNN_1D', 'SVM'),
    ('CNN_1D', 'CNN_LSTM'),
    ('CNN_1D', 'Ensemble'),
    ('LSTM', 'Ensemble'),
    ('CNN_LSTM', 'Ensemble'),
    ('RandomForest', 'Ensemble')
]

def load_predictions(model_name):
    path = os.path.join(OUTPUT_DIR, f'{model_name}_predictions.csv')
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return None, None
    df = pd.read_csv(path)
    return df['y_true'].values, df['y_pred'].values

def run_statistical_tests():
    print("\n📊 Running Statistical Tests...\n")
    results = []

    for model1, model2 in MODEL_PAIRS:
        y_true_1, y_pred_1 = load_predictions(model1)
        y_true_2, y_pred_2 = load_predictions(model2)

        if y_true_1 is None or y_true_2 is None:
            continue

        # Align lengths
        min_len = min(len(y_true_1), len(y_true_2))
        y_true_1, y_pred_1 = y_true_1[:min_len], y_pred_1[:min_len]
        y_true_2, y_pred_2 = y_true_2[:min_len], y_pred_2[:min_len]

        if not all(y_true_1 == y_true_2):
            print(f"⚠️ Label mismatch between {model1} and {model2}. Skipping.\n")
            continue

        correct_1 = (y_pred_1 == y_true_1).astype(int)
        correct_2 = (y_pred_2 == y_true_2).astype(int)

        # Paired t-test
        t_stat, p_val_t = ttest_rel(correct_1, correct_2)

        # McNemar's Test
        contingency = [[0, 0], [0, 0]]
        for a, b, label in zip(y_pred_1, y_pred_2, y_true_1):
            if a == label and b == label:
                contingency[0][0] += 1  # both correct
            elif a == label and b != label:
                contingency[0][1] += 1  # model1 correct, model2 wrong
            elif a != label and b == label:
                contingency[1][0] += 1  # model2 correct, model1 wrong
            else:
                contingency[1][1] += 1  # both wrong

        mcnemar_result = mcnemar(contingency, exact=False, correction=True)
        p_val_mcnemar = mcnemar_result.pvalue

        # Output to terminal
        print(f"🔍 {model1} vs {model2}")
        print(f"  📈 Paired t-test p-value      : {p_val_t:.5f}")
        print(f"  🧪 McNemar's test p-value     : {p_val_mcnemar:.5f}")
        print(f"  Contingency Table: {contingency}\n")

        # Store result
        results.append({
            'Model 1': model1,
            'Model 2': model2,
            'Paired t-test p-value': round(p_val_t, 5),
            "McNemar's test p-value": round(p_val_mcnemar, 5),
            'Both Correct': contingency[0][0],
            'Model1 Correct Only': contingency[0][1],
            'Model2 Correct Only': contingency[1][0],
            'Both Wrong': contingency[1][1]
        })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "statistical_tests_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved results to {csv_path}")

if __name__ == "__main__":
    run_statistical_tests()
