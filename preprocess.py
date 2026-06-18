"""
Preprocessing pipeline for the Ames Housing dataset.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_preprocess(filepath: str):
    """
    Load CSV, engineer features, and return X, y, and a fitted-ready preprocessor.

    Returns
    -------
    X : pd.DataFrame
    y : np.ndarray  (log1p of SalePrice)
    preprocessor : ColumnTransformer (unfitted — fit inside Pipeline)
    """
    df = pd.read_csv(filepath)

    # ── Feature engineering ─────────────────────────────────────────────────
    # Total square footage
    df["TotalSF"] = (
        df["GrLivArea"]
        + df.get("TotalBsmtSF", pd.Series(0, index=df.index)).fillna(0)
    )

    # House age at time of sale
    sale_year = 2010  # dataset collected up to 2010
    df["HouseAge"] = sale_year - df["YearBuilt"]

    # Years since last remodel
    if "YearRemodAdd" in df.columns:
        df["YearsSinceRemod"] = sale_year - df["YearRemodAdd"]

    # Bath score
    df["TotalBaths"] = df["FullBath"] + 0.5 * df.get("HalfBath", 0)

    # Drop columns that leak info or have too many nulls
    drop_cols = ["SalePrice"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Select features ──────────────────────────────────────────────────────
    numerical_features = [
        c for c in [
            "GrLivArea", "TotalSF", "HouseAge", "YearsSinceRemod",
            "TotalBsmtSF", "GarageArea", "LotArea",
            "OverallQual", "GarageCars", "TotalBaths",
        ] if c in df.columns
    ]

    categorical_features = [
        c for c in ["Neighborhood", "MSZoning", "KitchenQual"]
        if c in df.columns
    ]

    X = df[numerical_features + categorical_features].copy()

    # ── Target (log-transform removes right skew) ────────────────────────────
    target_col = pd.read_csv(filepath)["SalePrice"]
    y = np.log1p(target_col.values)

    # ── Sklearn preprocessing pipelines ─────────────────────────────────────
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    return X, y, preprocessor
