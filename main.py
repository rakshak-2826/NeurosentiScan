import os

def run_step(script_name, description):
    print(f"\n🚀 Step: {description}")
    exit_code = os.system(f"python scripts/{script_name}")
    if exit_code != 0:
        print(f"❌ Failed: {script_name}")
    else:
        print(f"✅ Done: {script_name}")

if __name__ == "__main__":
    print("\n🧠 Starting Full NeuroSentiScan Pipeline")

    # ML model training
    run_step("train_ml.py", "Train ML Models")

    # DL model training
    run_step("train_dl.py", "Train DL Models (CNN, LSTM, CNN+LSTM)")

    # DL Raw EEG models (EEGNet, BiLSTM, CNN_1D)
    run_step("train_dl_raw.py", "Train Raw EEG Models (EEGNet, BiLSTM, CNN_1D)")

    # Generate predictions from all trained models
    run_step("generate_predictions.py", "Generate Predictions")

    # Ensemble all predictions
    run_step("ensemble_predictions.py", "Ensemble Model Evaluation")

    # Explainability with LIME
    run_step("explainability.py", "Run LIME Interpretability")

    # PCA + t-SNE Visualization
    run_step("tsne_pca.py", "Run Dimensionality Reduction Visuals")

    # Plot Accuracy Bar Charts
    run_step("plot_results.py", "Generate Accuracy Comparison Plots")

    # Run Statistical Comparison
    run_step("stats_tests.py", "Run Statistical Tests (t-test + McNemar)")

    print("\n🎉 Full Pipeline Execution Completed.")
