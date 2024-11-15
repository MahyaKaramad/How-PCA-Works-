
# Comparison of Manual and Library-Based PCA Methods

This project demonstrates and compares two approaches to dimensionality reduction using Principal Component Analysis (PCA):
1. Manual Method: Implementing PCA using covariance matrix and eigenvector computations.
2. Library Method: Using the sklearn library for quick and easy PCA.

In this project, random data with 10 features is generated and then reduced to 2 dimensions for visualization.

## Table of Contents

- Introduction
- Requirements
- Installation and Setup
- Usage
- Code Overview
- Results
- License

## Introduction

Principal Component Analysis (PCA) is a popular method for reducing the dimensionality of data while retaining most of its variance. This project compares the results of dimensionality reduction using both manual and library-based methods, with plots to visualize and compare the outputs.

## Requirements

- Python 3.x
- Required libraries:
  - numpy
  - matplotlib
  - scikit-learn

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the code and view PCA dimensionality reduction results, use the following command:

```bash
python pca_comparison.py
```

## Code Overview

1. **Generate Random Data**: Creates a dataset with 100 samples and 10 features.
2. **Plot Original Data**: Displays the original data in two of its initial dimensions before applying PCA.
3. **Normalize Data**: Normalizes the data, which is essential for PCA.
4. **Manual PCA Implementation**:
   - Compute the covariance matrix.
   - Calculate eigenvalues and eigenvectors.
   - Sort eigenvalues and select the top components.
   - Reduce data dimensions to 2 principal components.
5. **PCA Using Library**: Uses sklearn’s PCA with 2 components.
6. **Plot Reduced Data**: Visualizes the reduced data for both methods.

## Results

1. Original Data (Before PCA)
2. Reduced Data (After Library PCA)
3. Reduced Data (After Manual PCA)

The plots demonstrate data reduction to 2 principal components, showing that both manual and library implementations produce similar results.

## License

This project is licensed under the MIT License.
