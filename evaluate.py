"""
House Price Prediction — Model Evaluation
Loads the trained model and produces evaluation plots.

Run: python src/evaluate.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split

DATA_PATH = "data/AmesHousing.csv"
MODEL_PATH = "models/rf_model.pkl"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true / 1000, y_pred / 1000, alpha=0.4, s=20, color="#4C72B0")
    lim = max(y_true.max(), y_pred.max()) / 1000
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($K)")
    ax.set_ylabel("Predicted Price ($K)")
    ax.set_title("Predicted vs Actual Sale Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_predicted_vs_actual.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/06_predicted_vs_actual.png")


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred / 1000, residuals / 1000, alpha=0.4, s=20, color="#55A868")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Price ($K)")
    axes[0].set_ylabel("Residual ($K)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals / 1000, bins=40, color="#55A868", edgecolor="white")
    axes[1].set_xlabel("Residual ($K)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_residuals.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/07_residuals.png")


def main():
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    X, y_log, _ = load_and_preprocess(DATA_PATH)
    _, X_test, _, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    y_pred_log = model.predict(X_test)

    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    print("\nEvaluation plots saved to plots/")


if __name__ == "__main__":
    main()
