# Computer Vision: Face Detection and Matching

## Authors

- Ibrahim Fateen  
- Bassant Zaki  
- Mohamed Hisham  
- Yasmine Mahmoud  

## Overview

This project focuses on building a facial recognition system using classical computer vision techniques. The core pipeline involves face detection, data preprocessing, dimensionality reduction, classification, and evaluation.

Key components include:
1. Dataset cleaning and preparation
2. Face detection using Haar Cascades
3. Feature extraction via PCA (Principal Component Analysis)
4. Face recognition using K-Nearest Neighbors (KNN)
5. ROC curve implementation for evaluation

## Dataset

We used a subset of the **FEI Face Database**, which contains images of 50 individuals. A specific subset of 10 individuals was chosen, and for each subject, 5 fixed poses were selected for training, while the remaining poses were used for testing.

## Face Detection

The system uses OpenCV’s Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) for detecting faces.

Detection steps:
- Convert image to grayscale
- Apply Haar classifier with appropriate parameters
- Crop the detected face
- Resize to 100x100 pixels

Advantages:
- Fast and lightweight
- Easily integrated via OpenCV

Limitations:
- Sensitive to lighting and pose variation
- Less accurate than modern CNN-based methods

## PCA (Principal Component Analysis)

PCA was implemented from scratch using NumPy.

Steps:
- Flatten all face images into vectors
- Center the dataset by subtracting the mean face
- Compute the covariance matrix or use SVD for efficiency
- Perform eigen decomposition to obtain eigenfaces
- Project faces into eigenspace for dimensionality reduction

Optimization:
- Switched from full covariance to `np.linalg.svd()` for faster computation

## Face Recognition

Recognition is performed by a **K-Nearest Neighbors (KNN)** classifier.

Steps:
1. Detect and preprocess faces
2. Reduce dimensionality with PCA
3. Train KNN on eigenface vectors

Unknown face detection:
- Implemented thresholding on KNN distances to reject unknown subjects

## Data Augmentation

To improve model robustness, the training dataset was augmented with:
- Horizontal flips
- Small rotations (±10°)
- Brightness adjustments

This increased the model’s tolerance to pose and lighting variations.

## Evaluation

The system achieved approximately 90% accuracy on the filtered test dataset. To further assess performance:

- ROC curve implemented from scratch using NumPy
- Compared results with scikit-learn’s ROC functions
- Measured AUC, TPR, FPR
- Supported two thresholding methods: Youden’s index and closest-to-(0,1)

## ROC Module

Custom class `ROCCurve` includes:
- `compute_roc()`: Generates TPR, FPR
- `_compute_auc()`: Calculates AUC using trapezoidal rule
- `optimal_threshold()`: Determines best threshold
- `plot_roc_curve()`: Visualizes ROC curve
- `get_performance_metrics()`: Outputs precision metrics at chosen threshold

Also includes `roc_curve_from_scratch()` as a simplified interface.

## Dependencies

- Python 3.8 or higher
- NumPy
- OpenCV
- scikit-learn
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Conclusion

This task demonstrates a complete pipeline for face detection and recognition using classical computer vision tools. With well-structured preprocessing, PCA, and simple classifiers like KNN, high performance can be achieved even with limited data. The inclusion of custom evaluation tools further enhances system interpretability.
