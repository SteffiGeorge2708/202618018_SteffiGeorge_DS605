# DS605 Lab 4: End-to-End NYC Airbnb Price Prediction Pipeline

## 📌 Executive Summary
This repository contains a complete, end-to-end Machine Learning pipeline designed to predict nightly rental prices for Airbnb listings across New York City using the Kaggle dataset (`AB_NYC_2019.csv`). The solution integrates data preprocessing, outlier removal, cross-validated model evaluation, model serialization, and an interactive Streamlit application.

---

## 🧹 Task 1: Data Analysis and Preparation
- **Outlier Cleaning:** Filtered listings to prices between **$10 and $500/night** and minimum night stays $\le 30$ days to prevent extreme skewness.
- **Missing Value Imputation:** Numerical missing values (`reviews_per_month`) imputed using column medians; categorical features imputed using most frequent values.
- **Encoding & Scaling:** Categorical variables (`room_type`, `neighbourhood_group`) encoded via `OneHotEncoder`. Continuous numeric features scaled using `StandardScaler`.
- **Target Transformation:** Applied `log1p` transformation to the `price` column to normalize skewed target distributions.

---

## 📊 Task 2: Model Training and Evaluation
Three regression models were evaluated using an 80/20 train-test split. Metrics were evaluated on actual dollar values (`expm1`):

| Model | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- |
| Linear Regression | $43.94 | $69.35 | 0.3940 |
| Decision Tree | $41.27 | $65.77 | 0.4549 |
| **Random Forest (Best Model)** | **$39.02** | **$62.56** | **0.5069** |

- **Best Model:** Random Forest Regressor captured non-linear spatial and room interactions best.
- **Pipeline Export:** The full preprocessing and model pipeline was serialized to `models/airbnb_pipeline.pkl`.

---

## 💻 Task 3: Streamlit Application
An interactive Streamlit interface (`202618018_lab_4.py`) was built featuring:
- High-contrast typography and clear parameter inputs in the sidebar.
- Instantaneous price estimation using the loaded model.
- Visual market distribution plots and feature importance rankings.

---

## ⚠️ Task 4: System Limitations
1. **Unobserved Listing Features:** Key property attributes like amenities (WiFi, pool, AC) and host review scores were absent from the dataset.
2. **Temporal Dynamics:** The dataset lacks seasonal pricing factors (peak holiday rates vs. off-peak rates).
3. **Coarse Spatial Modeling:** Relies solely on latitude/longitude coordinates rather than micro-location factors like proximity to subway stations.