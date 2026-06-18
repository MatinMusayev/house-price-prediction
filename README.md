# House Price Prediction

A machine learning project that predicts house sale prices using the Ames Housing dataset. Built with scikit-learn, pandas, and matplotlib.

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Ridge (baseline) | 0.91 | $31,071 | $41,508 |
| Random Forest | 0.89 | $32,382 | $42,582 |

5-fold CV R²: **0.889 ± 0.010**

## Project structure

```
house-price-prediction/
├── data/
│   └── AmesHousing.csv       # Dataset (1,460 houses, 17 features)
├── models/
│   └── rf_model.pkl          # Trained model (generated after training)
├── plots/                    # EDA and evaluation charts (auto-generated)
├── src/
│   ├── preprocess.py         # Feature engineering + sklearn pipeline
│   ├── eda.py                # Exploratory data analysis + plots
│   ├── train.py              # Model training + cross-validation
│   ├── evaluate.py           # Evaluation plots (predicted vs actual, residuals)
│   └── predict.py            # Inference on new data
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/MatinMusayev/house-price-prediction
cd house-price-prediction
pip install -r requirements.txt

# 1. Explore the data
python src/eda.py

# 2. Train the model
python src/train.py

# 3. Evaluate
python src/evaluate.py

# 4. Predict on a sample house
python src/predict.py

# Predict on your own CSV
python src/predict.py --input data/your_houses.csv
```

## Pipeline

**1. EDA** — distribution of SalePrice (right-skewed → log transform applied), correlation with numerical features, scatter plots.

**2. Feature engineering**
- `TotalSF` = GrLivArea + TotalBsmtSF
- `HouseAge` = 2010 − YearBuilt
- `YearsSinceRemod` = 2010 − YearRemodAdd
- `TotalBaths` = FullBath + 0.5 × HalfBath

**3. Preprocessing** — median imputation for numerical features, mode imputation for categorical, StandardScaler, OneHotEncoder. All inside a scikit-learn `Pipeline` to prevent data leakage.

**4. Modeling** — Ridge regression as baseline, Random Forest as the main model. Tuned `n_estimators`, `max_depth`, `min_samples_split`.

**5. Evaluation** — R², MAE, RMSE on a held-out 20% test set. 5-fold cross-validation on the training set. Residual analysis.

## Top features by importance

| Feature | Importance |
|---|---|
| OverallQual | 0.538 |
| GrLivArea | 0.312 |
| TotalSF | 0.070 |
| HouseAge | 0.012 |
| LotArea | 0.012 |

## Tech stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib
