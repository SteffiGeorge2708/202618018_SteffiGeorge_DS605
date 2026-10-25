# Lab 8: Decision Trees and Support Vector Machines (SVMs)

**Name:** Steffi George  
**Roll Number:** 202618018  
**Assignment:** Lab Assignment 8

## Objective

The objective of this lab is to understand and implement Decision Trees and Support Vector Machines (SVMs). The lab covers implementing algorithms from scratch, comparing them with Scikit-learn implementations, studying different SVM kernels, and evaluating model performance using classification metrics.

## Dataset

**Dataset:** Banknote Authentication Dataset

The dataset contains 1,372 samples and five columns.

- `variance`: Variance of the wavelet-transformed image
- `skewness`: Skewness of the wavelet-transformed image
- `curtosis`: Curtosis of the wavelet-transformed image
- `entropy`: Entropy of the wavelet-transformed image
- `class`: Target label (0 or 1)

The dataset contains 762 samples of class 0 and 610 samples of class 1. No missing values were found.

For Parts A and B, two features (`variance` and `skewness`) were used to visualize decision boundaries. Part D used all four input features.

The train-test split was 80:20, with `random_state=42` and stratification.

## Part A: Decision Tree from Scratch

### Implementation

A Decision Tree classifier was implemented from scratch using Python and NumPy. The implementation includes:

- Gini impurity and entropy calculations
- Finding the best feature and threshold for splitting
- Recursive tree construction
- Prediction using the constructed tree
- Decision boundary visualization
- Comparison with Scikit-learn's `DecisionTreeClassifier`
- Experiments with Gini and entropy criteria at different maximum depths

### Results

Using the two features `variance` and `skewness`, with Gini impurity and maximum depth 5:

| Metric | Manual Decision Tree | Scikit-learn Decision Tree |
|---|---:|---:|
| Accuracy | 93.09% | 93.09% |
| Precision | 88.15% | 88.15% |
| Recall | 97.54% | 97.54% |
| F1-score | 92.61% | 92.61% |
| Training time | 1.8099 s | 0.0132 s |
| Prediction time | 0.0021 s | 0.0020 s |

The prediction agreement between the manual implementation and Scikit-learn was 100% for this configuration.

### Hyperparameter Experiment

The manual Decision Tree was tested with Gini and entropy criteria and maximum depths of 2, 3, 5, 8, and unlimited depth.

The best F1-score in this experiment was **92.86%**, achieved using entropy with maximum depth 5. This setting achieved 93.45% accuracy.

### Observations

- Gini impurity and entropy both helped identify useful splits.
- Increasing tree depth improved performance up to a point, but deeper trees did not always produce better results.
- The manual implementation matched Scikit-learn's predictions for the tested depth-5 Gini configuration.
- Scikit-learn trained considerably faster than the manual implementation.
- Decision Trees are interpretable because their predictions can be represented using feature-based rules.

## Part B: Linear SVM from Scratch

### Implementation

A linear SVM was implemented from scratch using NumPy and subgradient descent. The implementation uses a regularized hinge-loss objective.

The following tasks were performed:

- Feature scaling using `StandardScaler`
- Converting class labels from 0 and 1 to -1 and +1
- Training a manual linear SVM
- Plotting the training loss
- Visualizing the decision boundary, margins, and margin violations
- Comparing the manual model with Scikit-learn's linear SVM
- Experimenting with different values of `C`, learning rates, and iteration counts

### Results

For the manual SVM with `C=1`, learning rate `0.001`, and 5,000 iterations:

| Metric | Manual Linear SVM | Scikit-learn Linear SVM |
|---|---:|---:|
| Accuracy | 86.55% | 87.64% |
| Precision | 91.26% | 84.92% |
| Recall | 77.05% | 87.70% |
| F1-score | 83.56% | 86.29% |
| Training time | 1.0292 s | 0.0809 s |
| Prediction time | 0.0011 s | 0.0096 s |

### Hyperparameter Experiments

The manual SVM was evaluated with different values of `C`.

- `C = 0.01`: Accuracy 85.82%, F1-score 81.34%
- `C = 0.1`: Accuracy 85.82%, F1-score 81.34%
- `C = 1`: Accuracy 86.55%, F1-score 83.56%
- `C = 10`: Accuracy 88.73%, F1-score 86.92%
- `C = 100`: Accuracy 88.36%, F1-score 86.78%

The highest F1-score in the `C` experiment was **86.92%**, achieved with `C=10`.

The learning-rate and iteration experiments also showed that training settings affect the final performance. Among the tested combinations, a learning rate of `0.0001` with 5,000 iterations achieved an F1-score of 87.50%.

### Observations

- A linear SVM learns a straight decision boundary.
- Feature scaling is important because SVM optimization depends on feature magnitudes.
- The regularization parameter `C` affects the trade-off between classification errors and the margin.
- Learning rate and iteration count influence convergence and performance.
- Scikit-learn's implementation trained faster than the manual implementation in the main comparison.

## Part C: Kernel SVM and Nonlinear Boundaries

### Implementation

A nonlinear dataset was generated using `make_moons` with 500 samples, noise `0.20`, and `random_state=42`. The dataset was divided into 400 training samples and 100 testing samples.

Three SVM kernels were evaluated:

- **Linear kernel:** Creates a straight decision boundary.
- **Polynomial kernel:** Can model curved decision boundaries.
- **RBF kernel:** Can model more complex nonlinear boundaries.

The models were compared using accuracy, precision, recall, F1-score, training time, and prediction time.

### Results

| Kernel | Accuracy | Precision | Recall | F1-score | Training time |
|---|---:|---:|---:|---:|---:|
| Linear | 86.00% | 84.62% | 88.00% | 86.27% | 0.0632 s |
| Polynomial | 89.00% | 83.05% | 98.00% | 89.91% | 0.0076 s |
| RBF | 99.00% | 100.00% | 98.00% | 98.99% | 0.0060 s |

### Observations

- The linear kernel achieved 86% accuracy but could not fully capture the curved structure of the dataset.
- The polynomial kernel improved accuracy to 89% and achieved 98% recall.
- The RBF kernel performed best, achieving 99% accuracy and a 98.99% F1-score.
- The RBF kernel was also the fastest to train among these three models in this experiment.
- Nonlinear kernels can perform better when the classes cannot be separated effectively by a straight line.

## Part D: Final Model Comparison

### Implementation

All five models were evaluated using the four original Banknote Authentication features: `variance`, `skewness`, `curtosis`, and `entropy`.

The models compared were:

1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

The training set contained 1,097 samples, and the test set contained 275 samples. `StandardScaler` was fitted only on the training features and then applied to the test features.

### Final Results

| Model | Accuracy | Precision | Recall | F1-score | Training time |
|---|---:|---:|---:|---:|---:|
| Scikit-learn RBF SVM | 100.00% | 100.00% | 100.00% | 100.00% | 0.0113 s |
| Manual Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 4.7155 s |
| Scikit-learn Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 0.0486 s |
| Scikit-learn Linear SVM | 97.45% | 94.57% | 100.00% | 97.21% | 0.0331 s |
| Manual Linear SVM | 88.00% | 92.38% | 79.51% | 85.46% | 1.3452 s |

*Training and prediction times are the measured times from this notebook run and may vary between runs and systems.*

### Observations

- The Scikit-learn RBF SVM achieved the highest F1-score of 100% and the highest accuracy of 100% on the test set.
- The manual and Scikit-learn Decision Trees both achieved 99.27% accuracy and a 99.19% F1-score.
- The Scikit-learn Decision Tree trained much faster than the manual Decision Tree.
- The Scikit-learn Linear SVM outperformed the manual Linear SVM in accuracy, recall, and F1-score.
- The manual Linear SVM had the longest prediction recall gap, with recall of 79.51%, indicating that it missed more positive-class samples than the other models.
- The RBF SVM provided the best overall test-set performance in this comparison. However, perfect test performance on this split does not guarantee perfect performance on unseen data.

## Conclusion

This lab provided practical experience with Decision Trees and Support Vector Machines. The manual implementations helped demonstrate how impurity-based splitting and hinge-loss optimization work. Comparing them with Scikit-learn highlighted the benefits of optimized library implementations.

The experiments showed that Decision Trees are useful when interpretability is important, while SVM performance depends on feature scaling, hyperparameters, and kernel choice. In the nonlinear `make_moons` experiment, the RBF kernel performed best. In the final comparison using all four Banknote Authentication features, the Scikit-learn RBF SVM achieved the highest test-set performance.

## Files

- `202618018_lab_8.ipynb` — Notebook containing implementations, experiments, visualizations, and results.
- `data_banknote_authentication.txt` — Banknote Authentication dataset.
- `part_d_final_comparison.csv` — Final model comparison results generated by the notebook.
- `README.md` — Description of the lab, methods, results, and conclusions.
