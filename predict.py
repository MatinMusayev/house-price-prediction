"""
House Price Prediction — Inference Script
Load a trained model and predict on new data.

Usage:
  python src/predict.py
  python src/predict.py --input data/new_houses.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "models/rf_model.pkl"


def predict_from_dict(house: dict, model) -> float:
    """Predict price for a single house given as a dictionary."""
    df = pd.DataFrame([house])

    # Derived features
    sale_year = 2010
    df["TotalSF"] = df.get("GrLivArea", 0) + df.get("TotalBsmtSF", pd.Series([0])).fillna(0)
    df["HouseAge"] = sale_year - df.get("YearBuilt", sale_year)
    df["YearsSinceRemod"] = sale_year - df.get("YearRemodAdd", df.get("YearBuilt", sale_year))
    df["TotalBaths"] = df.get("FullBath", 1) + 0.5 * df.get("HalfBath", pd.Series([0]))

    log_price = model.predict(df)[0]
    return np.expm1(log_price)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to CSV with houses to predict")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}\n")

    if args.input:
        df = pd.read_csv(args.input)
        # Derived features
        df["TotalSF"] = df["GrLivArea"] + df.get("TotalBsmtSF", 0).fillna(0)
        df["HouseAge"] = 2010 - df["YearBuilt"]
        df["YearsSinceRemod"] = 2010 - df.get("YearRemodAdd", df["YearBuilt"])
        df["TotalBaths"] = df["FullBath"] + 0.5 * df.get("HalfBath", 0)

        preds = np.expm1(model.predict(df))
        df["PredictedPrice"] = preds.round(0).astype(int)
        print(df[["PredictedPrice"]].to_string())
    else:
        # Demo prediction on a sample house
        sample = {
            "GrLivArea": 1800,
            "TotalBsmtSF": 900,
            "OverallQual": 7,
            "YearBuilt": 1995,
            "YearRemodAdd": 2005,
            "FullBath": 2,
            "HalfBath": 1,
            "GarageArea": 400,
            "GarageCars": 2,
            "LotArea": 8000,
            "Neighborhood": "CollgCr",
            "MSZoning": "RL",
            "KitchenQual": "Gd",
            "BedroomAbvGr": 3,
        }
        price = predict_from_dict(sample, model)
        print("Sample house prediction")
        print("-" * 30)
        for k, v in sample.items():
            print(f"  {k:<20}: {v}")
        print(f"\n  Predicted Price: ${price:,.0f}")


if __name__ == "__main__":
    main()
