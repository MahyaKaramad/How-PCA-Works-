PCA Dimensionality Reduction - Manual and Library Implementation

This project demonstrates the use of Principal Component Analysis (PCA) for dimensionality reduction. The code includes two implementations of PCA:

Manual PCA: Using numpy for eigenvalue and eigenvector computation.
Library PCA: Using sklearn's PCA for simplicity and speed.
The data is randomly generated with 10 features and is then reduced to 2 dimensions for visualization.

Table of Contents
Introduction
Requirements
Installation
Usage
Code Overview
Results
License
Introduction
Principal Component Analysis (PCA) is a popular method for reducing the dimensionality of datasets while retaining most of the variance. This project applies PCA in two ways: manually calculating the covariance matrix and eigenvalues, and using the sklearn PCA library. The results of both approaches are then plotted for visual comparison.

Requirements
Python 3.x
Required libraries:
numpy
matplotlib
scikit-learn

Code Overview
Generate Random Data: Creates a dataset with 100 samples and 10 features.
Plot Original Data: Visualizes the original dataset in two of its original dimensions.
Normalize Data: Normalizes the data, essential for PCA.
Manual PCA:
Calculate the covariance matrix.
Compute eigenvalues and eigenvectors.
Sort eigenvalues and select the top components.
Reduce the data dimensions to 2.
PCA Using Library: Uses sklearn’s PCA with 2 components.
Plot Reduced Data: Visualizes the reduced data for both methods.

# How-PCA-Works
