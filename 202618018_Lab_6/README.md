# Machine Learning Lab 6

## Student Details

* **Name:** Steffi George
* **Student ID:** 202618018
* **Course:** MSc Data Science
* **Lab:** 6

---

## Objective

This lab focuses on applying traditional machine learning techniques to **image and text data**.

The main objectives are:

* Extract meaningful features from images using traditional image-processing techniques.
* Perform text preprocessing and text vectorization.
* Train machine learning models for classification.
* Evaluate model performance.
* Compare different feature representations.
* Improve the representation and study the trade-off between feature dimensionality, computation time, and predictive performance.

**Deep learning, CNNs, and pretrained image embeddings are not used.**

---

# Part A – Image Feature Extraction

## Objective

The objective of Part A is to extract useful features from images using traditional image-processing techniques instead of deep learning.

## Method

The following steps were performed:

1. Loaded the sample images.
2. Converted images into suitable numerical representations.
3. Applied image preprocessing.
4. Used **Canny edge detection** to identify important edges in the images.
5. Extracted numerical features from the processed images.
6. Created a feature table for machine learning.

### Canny Edge Detection

Canny edge detection was used to identify boundaries and important structural information in the images.

Different Canny threshold values were examined to observe how they affect the detected edges.

The resulting edge images were visualized and compared.

## Output

The Part A outputs include:

* Sample image visualizations.
* Canny edge visualizations.
* Extracted image-feature table.

---

# Part B – Text Vectorization and Spam Classification

## Objective

The objective of Part B is to represent email text numerically and use a machine learning model to classify emails.

## Dataset

The email dataset contains word-frequency information and a `Prediction` column representing the target class.

The existing word-frequency columns were used to reconstruct the email text because the `clean_text` column did not contain usable text.

The following columns were not used as predictive text features:

* `Email No.` – identifier column.
* `Prediction` – target variable.
* `clean_text` – contained empty values.

## Text Representation

### CountVectorizer

`CountVectorizer` was used to convert text into numerical word-count features.

Each row represents an email and each generated feature represents a word or word combination.

## Classification Model

**Logistic Regression** was used as the classification model.

The workflow was:

```text
Email Data
    ↓
Text Preparation
    ↓
Train-Test Split
    ↓
CountVectorizer
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Evaluation
```

## Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Training time
* Prediction time
* Number of generated features

## TF-IDF

TF-IDF was not used in this implementation. The text representation comparison was performed using CountVectorizer and its improved representation.

---

# Part C – Improving the Text Representation

## Objective

The objective of Part C is to improve the text representation and examine its effect on model performance and computational requirements.

## Improvement

The CountVectorizer representation was improved by including **bigrams**.

The original representation considers individual words:

```text
free
money
click
```

The improved representation considers both individual words and two-word combinations:

```text
free money
click here
```

This can help capture relationships between words that individual word counts may not represent.

## Comparison

The original and improved representations were compared using:

* Accuracy
* Number of features
* Training time
* Prediction time

The comparison results were saved as:

```text
part_b_c_comparison.csv
```

---

# Overall Workflow

```text
Raw Data
   ↓
Preprocessing
   ↓
Feature Extraction / Vectorization
   ↓
Train-Test Split
   ↓
Machine Learning Model
   ↓
Evaluation
   ↓
Representation Improvement
   ↓
Comparison
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* OpenCV / PIL
* Matplotlib
* Jupyter Notebook

---

# Key Observations

* Traditional image-processing techniques can be used to extract useful image features without using deep learning.
* Canny edge detection highlights important boundaries and structural information in images.
* CountVectorizer converts textual information into numerical features that can be used by machine learning algorithms.
* Logistic Regression can be applied to the resulting text representation for classification.
* Adding bigrams increases the number of generated features because combinations of two words are also represented.
* Increasing feature dimensionality can affect training and prediction time.
* The improved representation should be evaluated using both predictive performance and computational cost rather than accuracy alone.

---

# Files

The repository contains:

* Jupyter notebook / Python code for the complete assignment.
* Sample image and Canny-edge visualizations.
* Extracted image-feature table.
* Text classification and vectorization results.
* `part_b_c_comparison.csv`
* This README file.

---

# Conclusion

This lab demonstrates a traditional machine learning workflow for both image and text data. Image features were obtained using classical image-processing techniques, while email data was converted into numerical representations using CountVectorizer. Logistic Regression was then used for classification, and an improved n-gram representation was evaluated to study the effect of feature representation on predictive performance and computational requirements.

No CNN, deep-learning model, or pretrained image embedding was used.
