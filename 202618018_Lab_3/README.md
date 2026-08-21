# Scikit-learn Preprocessing and Classification

## Assignment Title

**Scikit-learn: Data Preprocessing and Model Performance Evaluation**

## Student Information

* **Name:** Steffi George
* **ID:** 202618018

## Dataset

**Dataset:** Hotel Bookings Dataset

**Dataset Link:** `https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand`

The dataset contains hotel booking information and is used to predict whether a booking was canceled using the `is_canceled` variable as the target.

## Objective

The objective of this assignment is to build and compare two Scikit-learn preprocessing pipelines and evaluate two classification models.

## Preprocessing Choices

### Missing Values

* Missing values were checked for every column using both count and percentage.
* Columns with very high missingness were identified.
* The `company` column was removed because it contained a very high proportion of missing values.
* Numerical missing values were handled using `KNNImputer(n_neighbors=5)`.
* Categorical missing values were handled using `SimpleImputer(strategy="most_frequent")`.

### Data Leakage

The following columns were removed because they directly reveal the final booking outcome:

* `reservation_status`
* `reservation_status_date`

### Categorical Features

Categorical features were converted into numerical form using:

* `OneHotEncoder(handle_unknown="ignore")`

### Numerical Features

Two preprocessing pipelines were created:

**Pipeline A**

* `KNNImputer(n_neighbors=5)`
* `StandardScaler`

**Pipeline B**

* `KNNImputer(n_neighbors=5)`
* `MinMaxScaler`

### Train-Test Split

The dataset was split using:

* `test_size=0.2`
* `stratify=y`
* `random_state=42`

The same train-test split was used for all model comparisons.

## Models

Two classification models were evaluated with both preprocessing pipelines:

1. Logistic Regression with Pipeline A
2. Logistic Regression with Pipeline B
3. Decision Tree with Pipeline A
4. Decision Tree with Pipeline B

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

## Evaluation Metrics

The models were evaluated using:

* Training Accuracy
* Testing Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Final Observations

1. The best overall preprocessing-model combination was determined based on the testing performance, particularly the F1-score.

2. The performance of Logistic Regression was compared between StandardScaler and MinMaxScaler to determine whether the choice of scaling method affected its results.

3. Scaling had little major effect on the Decision Tree because decision trees are generally not sensitive to the scale of numerical features.

4. The difference between training and testing accuracy was used to identify possible overfitting. A large difference indicates that the model may be overfitting the training data.

5. The confusion matrices were used to compare correct predictions with false positives and false negatives and to further assess the classification performance.

## Conclusion

The four model-pipeline combinations were compared using the same train-test split and evaluation metrics. The final comparison shows which preprocessing method and classification model provide the most suitable performance for predicting hotel booking cancellations.
