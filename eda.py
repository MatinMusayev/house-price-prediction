"""
House Price Prediction — Exploratory Data Analysis
Generates charts saved to the plots/ folder.

Run: python src/eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DATA_PATH = "data/AmesHousing.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["SalePrice"] / 1000, bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("SalePrice distribution (raw)")
    axes[0].set_xlabel("Price ($K)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(df["SalePrice"]), bins=40, color="#55A868", edgecolor="white")
    axes[1].set_title("SalePrice distribution (log-transformed)")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_price_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/01_price_distribution.png")


def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
    top = corr.head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0" if v > 0 else "#C44E52" for v in top.values]
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title("Top 10 features correlated with SalePrice")
    ax.set_xlabel("Pearson correlation")
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_correlation.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/02_correlation.png")


def plot_price_vs_area(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["GrLivArea"], df["SalePrice"] / 1000,
               alpha=0.4, s=20, color="#4C72B0")
    ax.set_title("Living Area vs Sale Price")
    ax.set_xlabel("Above-ground living area (sq ft)")
    ax.set_ylabel("Sale Price ($K)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_area_vs_price.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/03_area_vs_price.png")


def plot_quality_vs_price(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="SalePrice", by="OverallQual", ax=ax,
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", color="#4C72B0", alpha=0.5),
               medianprops=dict(color="white", linewidth=2),
               whiskerprops=dict(color="#4C72B0"),
               capprops=dict(color="#4C72B0"),
               flierprops=dict(marker="o", markersize=2, alpha=0.3, color="gray"))
    plt.suptitle("")
    ax.set_title("Overall Quality vs Sale Price")
    ax.set_xlabel("Overall Quality (1–10)")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_quality_vs_price.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/04_quality_vs_price.png")


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
    ax.barh(missing.index[::-1], missing.values[::-1], color="#DD8452")
    ax.set_title("Missing values per column")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_missing_values.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/05_missing_values.png")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nSalePrice stats:\n{df['SalePrice'].describe().round(0)}")

    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_vs_area(df)
    plot_quality_vs_price(df)
    plot_missing_values(df)

    print("\nEDA complete. Check the plots/ folder.")


if __name__ == "__main__":
    main()
