# DS605 Lab Assignment 5

## Student Details

- Name: Steffi George
- Student ID: 202618018
- Course: DS605 - Fundamentals of Machine Learning

## Dataset

UCI Productivity Prediction of Garment Employees.

The dataset contains information about garment production activities,
including department, team, overtime, incentives, workers and
productivity.

## Tasks

### Part A - Scikit-learn

Implemented:

- Data preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Linear Regression
- Logistic Regression
- Regression and classification evaluation
- Training and prediction time measurement

### Part B - From Scratch

Implemented using NumPy and Pandas:

- Missing value handling
- Categorical encoding
- Feature scaling
- Linear Regression
- Logistic Regression
- Sigmoid function
- Gradient descent
- Predictions
- Evaluation metrics

### Part C - Comparison

Compared the Scikit-learn and from-scratch implementations using:

- MAE
- RMSE
- R²
- Accuracy
- Precision
- Recall
- F1 Score
- Training time
- Prediction time

An optimized from-scratch Logistic Regression implementation was
also tested using a different learning rate and number of iterations.

## Files

- `Lab5_Sklearn.ipynb` - Scikit-learn implementation
- `Lab5_From_Scratch.ipynb` - NumPy/Pandas implementation
- `Lab5_Comparison.ipynb` - Comparison and optimization
- `data/garments_worker_productivity.csv` - Dataset
- `sklearn_results.csv` - Scikit-learn results
- `manual_results.csv` - From-scratch results
- `comparison_results.csv` - Final comparison