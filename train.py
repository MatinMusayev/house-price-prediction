"""
House Price Prediction — Training Script
Trains a Random Forest model on the Ames Housing dataset.
Run: python src/train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from preprocess import load_and_preprocess

DATA_PATH = "data/AmesHousing.csv"
MODEL_PATH = "models/rf_model.pkl"


def evaluate(model, X_test, y_test, label="Model"):
    preds = model.predict(X_test)
    preds_exp = np.expm1(preds)
    y_exp = np.expm1(y_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_exp, preds_exp)
    rmse = np.sqrt(mean_squared_error(y_exp, preds_exp))
    print(f"\n{label} — Test set results:")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    return r2, mae, rmse


def main():
    print("=" * 50)
    print("House Price Prediction — Training")
    print("=" * 50)

    # 1. Load & preprocess
    X, y, preprocessor = load_and_preprocess(DATA_PATH)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 3. Baseline — Ridge regression
    ridge_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10))
    ])
    ridge_pipe.fit(X_train, y_train)
    evaluate(ridge_pipe, X_test, y_test, label="Ridge (baseline)")

    # 4. Main model — Random Forest
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 5. Cross-validation
    print("\nRunning 5-fold cross-validation on Random Forest...")
    cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring="r2")
    print(f"  CV R² scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Final fit & evaluate
    rf_pipe.fit(X_train, y_train)
    evaluate(rf_pipe, X_test, y_test, label="Random Forest (final)")

    # 7. Feature importances
    rf_model = rf_pipe.named_steps["regressor"]
    cat_cols = rf_pipe.named_steps["preprocessor"] \
        .named_transformers_["cat"]["onehot"].get_feature_names_out()
    num_cols = rf_pipe.named_steps["preprocessor"] \
        .named_transformers_["num"].feature_names_in_
    all_features = list(num_cols) + list(cat_cols)

    importances = pd.Series(rf_model.feature_importances_, index=all_features)
    top10 = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 feature importances:")
    for feat, imp in top10.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<30} {bar} {imp:.4f}")

    # 8. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_pipe, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
