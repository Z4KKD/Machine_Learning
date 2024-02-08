# Iris Dataset Analysis with Supervised and Unsupervised Learning

This repository contains Python scripts `supervised.py` and `unsupervised.py` for analyzing the Iris dataset using supervised and unsupervised learning techniques, respectively.

## Supervised Learning (`supervised.py`)

The `supervised.py` script demonstrates the use of K-Nearest Neighbors (KNN) algorithm for supervised learning on the Iris dataset. It performs the following steps:

1. Import necessary libraries including NumPy, Matplotlib, and scikit-learn.
2. Load the Iris dataset using scikit-learn's `load_iris()` function.
3. Split the dataset into training and testing sets using `train_test_split()`.
4. Standardize the features using `StandardScaler`.
5. Train a KNN classifier with 3 neighbors.
6. Make predictions on the test set and evaluate the model's accuracy using `accuracy_score` and `confusion_matrix`.
7. Visualize a scatter plot of the first two features (sepal length vs sepal width) with color-coded classes.

## Unsupervised Learning (`unsupervised.py`)

The `unsupervised.py` script demonstrates the use of K-Means clustering algorithm for unsupervised learning on the Iris dataset. It performs the following steps:

1. Import necessary libraries including NumPy, Matplotlib, and scikit-learn.
2. Load the Iris dataset using scikit-learn's `load_iris()` function.
3. Standardize the features using `StandardScaler`.
4. Apply Principal Component Analysis (PCA) for dimensionality reduction to visualize the clusters.
5. Perform K-Means clustering with 3 clusters.
6. Visualize the clusters in the reduced feature space.

## Usage

1. Clone or download the repository to your local machine.
2. Ensure you have Python installed along with the necessary libraries.
3. Navigate to the directory containing the Python scripts.
4. Run the `supervised.py` script for supervised learning and `unsupervised.py` script for unsupervised learning.

```bash
python supervised.py
python unsupervised.py
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
