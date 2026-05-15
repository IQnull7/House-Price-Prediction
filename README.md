# Ames Housing Price Prediction

End-to-end regression pipeline on the Ames Housing dataset, covering exploratory data analysis, preprocessing, feature engineering, model comparison, and diagnostic evaluation. Includes a comparative study on the effect of feature engineering and encoding strategy on model performance.

---

## Project Structure

```
├── data/
│   ├── train.csv                        # Kaggle half-dataset (1460 rows)
│   ├── test.csv                         # Kaggle test set (no SalePrice)
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb           # Cleaning + FE on half dataset
│   ├── 03_Models.ipynb                  # Model training on half dataset
│   ├── 04_Evaluation.ipynb              # Diagnostic evaluation on half dataset
│   ├── Preprocessing_complete_dataset.ipynb     # Cleaning + FE on full dataset
│   ├── Models_complete_dataset.ipynb            # Model training on full dataset
│   ├── Preprocessing_without_feature_engineering.ipynb  # No-FE preprocessing
│   └── Models_complete_data_no_engineering.ipynb        # Comparative study
├── src/
│   ├── preprocess.py
│   └── evaluate.py
├── plots/
├── requirements.txt
└── README.md
```

---

## Dataset

- **Half dataset:** Kaggle competition `train.csv` — 1460 rows, 80 features
- **Full dataset:** De Cock's original `AmesHousing.csv` — 2930 rows, 81 features
- **Target variable:** `SalePrice` (log-transformed during training)
- **After preprocessing:** 1308 / 2642 rows remaining after outlier removal

---

## Pipeline Overview

### 1. EDA (`01_EDA.ipynb`)
- Distribution analysis of numerical and categorical features
- Correlation analysis with `SalePrice`
- Missing value profiling
- Identification of skewed features and outliers

### 2. Preprocessing (`02_Preprocessing.ipynb`, `Preprocessing_complete_dataset.ipynb`)
- **Dtype fixes:** `GarageYrBlt`, `LotFrontage`, `MasVnrArea`, `MoSold`
- **Missing values:** Neighborhood-grouped median imputation for `LotFrontage`; `'None'` for feature-absence columns; mode for `Electrical`
- **Outlier removal:** IQR-based removal + manual cap on `GrLivArea` (>4000 sqft, price <$200k)
- **Skewness correction:** `log1p` for most skewed features; `sqrt` for `BsmtUnfSF`, `BsmtFinSF1`; `cbrt` for `TotalBsmtSF`
- **Feature engineering:**
  - Aggregations: `TotalSF`, `TotalBath`, `TotalPorchSF`
  - Binary flags: `HasGarage`, `HasFireplace`, `HasWoodDeck` etc.
  - Interaction terms: `LivArea_Qual`, `TotalSF_Qual`, `Bsmt_Qual`, `Bath_Qual`, `Garage_Qual`
  - Age features: `HouseAge`, `RemodAge`, `GarageAge`, `IsNew`
- **Encoding:** Ordinal mapping for quality/condition features; one-hot encoding for nominal categoricals

### 3. Modeling (`03_Models.ipynb`, `Models_complete_dataset.ipynb`)

Seven models trained and compared:

| Model | Half Dataset R² | Full Dataset R² |
|---|---|---|
| Simple Linear Regression | 0.7170 | 0.7525 |
| Polynomial Regression | 0.7155 | 0.7575 |
| Multiple Linear Regression | 0.8720 | 0.9098 |
| Decision Tree | 0.7604 | 0.7787 |
| Random Forest | 0.8404 | 0.9013 |
| XGBoost | 0.8616 | 0.9210 |
| Ridge (α=10) | 0.8765 | 0.9292 |
| **Lasso (α=0.001)** | **0.8758** | **0.9293** |

All models trained on log-transformed `SalePrice`. Hyperparameters tuned via 5-fold cross-validation.

### 4. Evaluation (`04_Evaluation.ipynb`)
- Residual analysis (scatter + distribution)
- Predicted vs actual plots
- Error breakdown by price segment (low / mid / high)
- 5-fold cross-validation stability
- Learning curves
- Feature importance (coefficients for linear models, impurity-based for tree models)
- Worst predictions analysis
- 90% prediction interval coverage

---

## Comparative Study: Feature Engineering vs Raw Data

A separate experiment (`Models_complete_data_no_engineering.ipynb`) trained the same models on cleaned data **without** feature engineering or one-hot encoding to isolate their effects.

| Model | With FE + OHE | No FE + Label Enc | Δ |
|---|---|---|---|
| Ridge | 0.9292 | 0.9129 | ↓ 0.016 |
| Lasso | 0.9293 | 0.9140 | ↓ 0.015 |
| XGBoost | 0.9210 | 0.9202 | ↓ 0.001 |
| Random Forest | 0.9013 | 0.8990 | ↓ 0.002 |

**Key finding:** Feature engineering had negligible impact on all models. The performance gap between linear and tree models was driven by **one-hot encoding** — linear models require it to correctly interpret nominal categoricals, while tree models are indifferent. Without one-hot encoding, XGBoost (0.9202) overtook Ridge (0.9129) and Lasso (0.9140), restoring the expected ranking.

---

## Key Results

- **Best model:** Lasso Regression (α=0.001) — R²=0.9293 on full dataset
- **Lasso feature selection:** 81/226 features zeroed out — the model identified 145 features as genuinely predictive
- **Top predictive features:** `LivArea_Qual`, `Bsmt_Qual`, `Bath_Qual`, `OverallQual`, `GarageCars`
- **Performance bottleneck on half dataset:** Feature-to-sample ratio (~1:5) constrained tree models more than linear models; switching to the full dataset improved Random Forest by +6% and XGBoost by +6.9%
- **Prediction interval:** Ridge achieves ~90% coverage with average interval width of ~$38,000

---

## Setup

```bash
git clone https://github.com/IQnull7/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

Download `train.csv` and `AmesHousing.csv` and place them in the `data/` folder before running notebooks.

- Kaggle dataset: [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Full Ames dataset: [AmesHousing on Kaggle](https://www.kaggle.com/datasets/marcopale/housing)

Run notebooks in order: `01_EDA` → `02_Preprocessing` → `03_Models` → `04_Evaluation`

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
