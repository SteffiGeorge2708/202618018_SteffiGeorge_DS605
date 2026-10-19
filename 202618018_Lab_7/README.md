# Lab Assignment 7: Image Representation, PCA, Eigenfaces and Face Recognition

**Name:** Steffi George  
**Roll Number:** 202618018  
**Assignment:** Lab Assignment 7

---

## 1. Overview

This project explores image representation, dimensionality reduction, eigenfaces, image reconstruction, compression, and face recognition using the **AT&T Database of Faces (ORL)**.

The face images are converted into grayscale pixel vectors and processed using Principal Component Analysis (PCA). PCA is implemented using NumPy linear algebra operations, including eigendecomposition and Singular Value Decomposition (SVD).

The resulting low-dimensional representations are used for similarity-based face matching and compared with traditional machine-learning classifiers.

## 2. Objectives

- Represent grayscale face images as numerical feature vectors.
- Create a reproducible train-test split with all subjects represented in both sets.
- Implement PCA using NumPy eigendecomposition and SVD.
- Visualize eigenfaces and cumulative explained variance.
- Reduce dimensionality while preserving at least 90% of the variance.
- Reconstruct face images and evaluate reconstruction quality.
- Compare the storage requirements of original and reduced representations.
- Perform face recognition using Euclidean distance.
- Compare raw-pixel matching, PCA-based matching, and traditional machine-learning classifiers.
- Investigate how the number of retained components affects recognition accuracy.

## 3. Dataset

**Dataset:** AT&T Database of Faces (ORL)

**Source:** [AT&T Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

The dataset contains 400 grayscale face images belonging to 40 subjects, with 10 images per subject.

Each image has dimensions of 112 × 92 pixels, giving 10,304 pixel features per image. Pixel intensity values in the loaded dataset range from 0 to 251.

### Dataset statistics

| Property | Value |
|---|---:|
| Total images | 400 |
| Number of subjects | 40 |
| Images per subject | 10 |
| Image dimensions | 112 × 92 |
| Pixel features per image | 10,304 |
| Minimum observed pixel intensity | 0 |
| Maximum observed pixel intensity | 251 |

## 4. Methodology

### Part A: Image Representation

The images were loaded and converted to grayscale using Pillow. Each image was flattened into a one-dimensional vector, and all vectors were combined into an image-feature matrix.

A stratified train-test split was performed using `random_state=42`, with 70% of the images assigned to training and 30% to testing.

| Property | Result |
|---|---:|
| Training images | 280 |
| Testing images | 120 |
| Subjects in training set | 40 |
| Subjects in testing set | 40 |
| Training matrix shape | (280, 10304) |
| Testing matrix shape | (120, 10304) |

This ensures that every subject is represented in both the training and testing sets.

### Part B: PCA, Eigenfaces and Compression

The mean face was calculated using the training images, and the training and testing data were centered using the training mean.

PCA was implemented using two approaches:

1. **Eigendecomposition:** The eigenvalues and eigenvectors were obtained from the image-space Gram matrix.
2. **Singular Value Decomposition (SVD):** NumPy's `numpy.linalg.svd()` was applied to the centered training matrix.

#### Comparison of PCA approaches

| Metric | Eigendecomposition | SVD |
|---|---:|---:|
| Principal directions obtained | 279 | 280 |
| Computation time | 1.0876 seconds | 6.1713 seconds |

The mean absolute similarity between the compared principal directions was **1.000000**. This indicates agreement between the approaches for the directions compared, allowing for possible sign differences in eigenvectors.

The difference in the number of directions arises because the centered training data can contain a zero-variance direction.

#### Explained variance

The minimum number of principal components required to preserve at least 90% of the variance was **92 components**.

| Metric | Result |
|---|---:|
| Original features per image | 10,304 |
| Retained PCA components | 92 |
| Cumulative variance preserved | 90.09% |
| Reduced training matrix | (280, 92) |
| Reduced testing matrix | (120, 92) |

The first principal component explained approximately 17.59% of the variance, while the first 10 components together explained approximately 60.32%.

#### Reconstruction quality

Sample test images were reconstructed using the retained PCA components.

| Metric | Result |
|---|---:|
| Reconstruction mean squared error (MSE) | 317.39 |
| Reconstruction root mean squared error (RMSE) | 17.82 |

The reconstruction errors quantify the difference between the original pixel values and the reconstructed images. The reconstruction figures provide a visual comparison of the information retained after dimensionality reduction.

#### Storage comparison

The notebook compared the number of stored numerical values and their corresponding storage requirements, assuming 8 bytes per value.

| Metric | Original representation | PCA representation |
|---|---:|---:|
| Number of stored values | 2,885,120 | 25,760 |
| Storage for feature values | 23.08 MB | 0.21 MB |

The reduced representation requires approximately **99.11% fewer numerical values** than the original training pixel representation.

This comparison covers the stored feature values and does not include the mean face, principal-component matrix, labels, or other model metadata required to reconstruct or use the PCA representation.

### Part C: Recognition in PCA Space

Face recognition was performed using nearest-neighbor matching with Euclidean distance. Each test image was compared with the stored training faces, and the label of the closest training face was assigned to the test image.

The recognition accuracy was compared between the original pixel space and the reduced PCA space.

| Recognition method | Accuracy |
|---|---:|
| Raw-pixel nearest neighbor | 98.33% |
| PCA nearest neighbor using 92 components | 98.33% |

Both methods correctly classified 118 of the 120 test images.

The results show that PCA-based recognition achieved the same accuracy as raw-pixel nearest-neighbor matching while using a substantially smaller feature representation.

Correct and incorrect matches were also inspected visually to understand the recognition results.

### Part D: Traditional Machine Learning

Traditional machine-learning classifiers were trained using PCA features. Their performance was compared with the similarity-based recognition methods.

#### Classifier comparison

| Method | Accuracy | Training and prediction time |
|---|---:|---:|
| Support Vector Machine (SVM) | 100.00% | 0.0598 seconds |
| K-Nearest Neighbors (KNN) | 98.33% | 9.3955 seconds |
| Logistic Regression | 97.50% | 0.4014 seconds |
| PCA nearest neighbor | 98.33% | Not recorded in comparison table |
| Raw-pixel nearest neighbor | 98.33% | Not recorded in comparison table |

The SVM achieved the highest observed accuracy of 100% on the held-out test set. KNN achieved 98.33%, while Logistic Regression achieved 97.50%.

The reported classifier times are those measured by the notebook for its training and prediction procedure. They are environment-dependent and should not be interpreted as universal performance benchmarks.

#### Effect of the number of PCA components

The notebook also examined recognition accuracy with different numbers of retained components.

| Components | KNN | Logistic Regression | Nearest Neighbor | SVM |
|---:|---:|---:|---:|---:|
| 10 | 95.83% | 97.50% | 95.83% | 97.50% |
| 20 | 96.67% | 95.83% | 96.67% | 96.67% |
| 30 | 97.50% | 98.33% | 97.50% | 99.17% |
| 40 | 98.33% | 98.33% | 98.33% | 100.00% |
| 60 | 98.33% | 98.33% | 98.33% | 100.00% |
| 92 | 98.33% | 97.50% | 98.33% | 100.00% |

The SVM achieved 100% accuracy with 40, 60, and 92 components in these experiments. This shows that, for this split and configuration, the SVM reached its best observed accuracy with fewer components than the 92-component representation selected to preserve at least 90% of the variance.

The results also show that the best number of components can depend on the classifier and that preserving more variance does not necessarily produce higher classification accuracy for every model.

## 5. Implementation and Technologies

The PCA calculations were implemented directly using NumPy linear algebra rather than Scikit-learn's PCA implementation.

Technologies used:

- **Python:** Main programming language
- **NumPy:** Numerical operations, eigendecomposition, SVD, and PCA projections
- **Pandas:** Result tables and CSV exports
- **Matplotlib:** Image visualization, eigenfaces, explained-variance plots, and reconstruction examples
- **Pillow:** Image reading and grayscale conversion
- **Scikit-learn:** Train-test splitting, traditional classifiers, and evaluation metrics

## 6. Project Structure

```text
202618018_Lab_7/
├── 202618018_lab_7.ipynb
├── README.md
├── orl_faces/
│   ├── s1/
│   ├── s2/
│   └── ...
└── outputs/
    ├── sample_faces.png
    ├── mean_face.png
    ├── recognition_comparison.csv
    ├── component_tuning.csv
    └── other generated figures
```

The `outputs` directory contains generated visualizations and CSV result tables. The exact figures included depend on the cells executed and the files saved by the notebook.

## 7. How to Run

1. Clone or download the repository.
2. Install the required Python libraries:

   ```bash
   pip install numpy pandas matplotlib pillow scikit-learn jupyter
   ```

3. Obtain and extract the ORL dataset, following its distribution terms.
4. Place the dataset folder next to the notebook, or update `DATASET_DIR` to the correct location.
5. Open `202618018_lab_7.ipynb` in VS Code or Jupyter Notebook.
6. Select a Python environment containing the required libraries.
7. Run the notebook cells from top to bottom.
8. Review the generated plots, reconstruction examples, recognition comparison, and component-tuning results.

## 8. Conclusion and Observations

This assignment demonstrated the use of PCA and eigenfaces for representing face images in a lower-dimensional space. NumPy eigendecomposition and SVD produced closely matching principal directions, and 92 components preserved 90.09% of the variance in the training data.

Reducing the representation from 10,304 pixel features to 92 PCA features lowered the number of stored numerical values by approximately 99.11%. The reconstruction RMSE was 17.82, showing that the reduced representation retained much of the image information while introducing some reconstruction error.

Raw-pixel and PCA-based nearest-neighbor matching both achieved 98.33% recognition accuracy. Among the tested classifiers, SVM achieved 100% accuracy on the held-out test set, with 40 components already sufficient to reach that score in the component-tuning experiments.

Overall, the experiments illustrate the potential of PCA to reduce image dimensionality while retaining useful information for face recognition. They also show that recognition performance depends on the representation, number of components, and classifier selected.

