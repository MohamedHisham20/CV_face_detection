import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class ROCCurve:
    """
    A class for computing and visualizing Receiver Operating Characteristic (ROC) curves from scratch.
    """
    
    def __init__(self):
        self.fpr: List[float] = []  # False Positive Rate
        self.tpr: List[float] = []  # True Positive Rate
        self.thresholds: List[float] = []  # Thresholds used
        self.auc: float = 0.0  # Area Under Curve

        
    
    def compute_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the ROC curve points from true labels and predicted scores.
        
        Args:
            y_true: Binary ground truth labels (0 or 1)
            y_scores: Predicted scores or probabilities
            
        Returns:
            Tuple containing false positive rates, true positive rates, and thresholds
        """
        # Convert inputs to numpy arrays if they aren't already
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        
        # Input validation
        if len(y_true) != len(y_scores):
            raise ValueError("Length of y_true and y_scores must be equal")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true should only contain binary values (0 or 1)")
        
        # Get unique thresholds in descending order
        # Add a threshold higher than max score to ensure we start with all negatives
        # Add a threshold lower than min score to ensure we end with all positives
        unique_scores = np.unique(y_scores)
        thresholds = np.append(np.append(np.inf, unique_scores), -np.inf)
        thresholds = np.sort(thresholds)[::-1]  # Sort in descending order
        
        # Count positives and negatives in the true labels
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0:
            raise ValueError("No positive samples found in y_true")
        if n_neg == 0:
            raise ValueError("No negative samples found in y_true")
        
        # Initialize arrays for TPR and FPR
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        
        # Compute TPR and FPR for each threshold
        for i, threshold in enumerate(thresholds):
            y_pred = (y_scores >= threshold).astype(int)
            
            # True positives and false positives
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            # True Positive Rate (Sensitivity, Recall) and False Positive Rate
            tpr[i] = tp / n_pos if n_pos > 0 else 0
            fpr[i] = fp / n_neg if n_neg > 0 else 0
        
        # Store results
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        
        # Compute AUC using the trapezoidal rule
        self.auc = self._compute_auc(fpr, tpr)
        
        return fpr, tpr, thresholds
    
    def _compute_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Compute the Area Under the ROC Curve using the trapezoidal rule.
        
        Args:
            fpr: False Positive Rate values
            tpr: True Positive Rate values
            
        Returns:
            The area under the ROC curve (AUC)
        """
        # Sort by increasing FPR if not already sorted
        indices = np.argsort(fpr)
        sorted_fpr = fpr[indices]
        sorted_tpr = tpr[indices]
        
        # Compute AUC using the trapezoidal rule
        width = np.diff(sorted_fpr)
        heights = (sorted_tpr[1:] + sorted_tpr[:-1]) / 2
        
        return np.sum(width * heights)
    
    def optimal_threshold(self, method: str = 'youden') -> float:
        """
        Find the optimal threshold based on the specified method.
        
        Args:
            method: Method to determine optimal threshold
                   'youden': Youden's J statistic (maximizes tpr - fpr)
                   'closest': Closest to (0,1)
        
        Returns:
            The optimal threshold value
        """
        if not self.fpr.size or not self.tpr.size:
            raise ValueError("ROC curve must be computed first")
        
        if method == 'youden':
            # Youden's J statistic (J = Sensitivity + Specificity - 1 = TPR - FPR)
            j_scores = self.tpr - self.fpr
            best_idx = np.argmax(j_scores)
        elif method == 'closest':
            # Closest to (0,1)
            distances = np.sqrt((1 - self.tpr) ** 2 + self.fpr ** 2)
            best_idx = np.argmin(distances)
        else:
            raise ValueError("Method must be 'youden' or 'closest'")
        
        return self.thresholds[best_idx]
    
    def plot_roc_curve(self, ax: Optional[plt.Axes] = None, title: str = "ROC Curve") -> plt.Axes:
        """
        Plot the ROC curve.
        
        Args:
            ax: Matplotlib axes to plot on. If None, creates new axes.
            title: Title for the plot
            
        Returns:
            The matplotlib axes with the ROC curve
        """
        if not self.fpr.size or not self.tpr.size:
            raise ValueError("ROC curve must be computed first")
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(self.fpr, self.tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {self.auc:.3f})')
        
        # Plot random guessing line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guessing')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_performance_metrics(self, threshold: Optional[float] = None) -> dict:
        """
        Calculate various performance metrics at a specific threshold.
        
        Args:
            threshold: Threshold to compute metrics at.
                      If None, uses Youden's optimal threshold.
                      
        Returns:
            Dictionary of performance metrics
        """
        if not self.fpr.size or not self.tpr.size:
            raise ValueError("ROC curve must be computed first")
            
        if threshold is None:
            threshold = self.optimal_threshold(method='youden')
        
        # Find the index of the closest threshold value
        idx = np.argmin(np.abs(self.thresholds - threshold))
        
        # Get values at that threshold
        tpr = self.tpr[idx]  # Sensitivity, Recall
        fpr = self.fpr[idx]  # False Positive Rate
        tnr = 1 - fpr  # Specificity, True Negative Rate
        
        # Calculate precision (assuming we have a way to get actual predictions at this threshold)
        # For binary classification with a threshold:
        # precision = TP / (TP + FP) = TPR * P / (TPR * P + FPR * N)
        # Where P and N are the total positive and negative samples
        # Since we don't have P and N, we'll just include placeholders
        metrics = {
            'threshold': threshold,
            'tpr': tpr,  # Sensitivity, Recall
            'fpr': fpr,  # False Positive Rate
            'tnr': tnr,  # Specificity, True Negative Rate
            'auc': self.auc  # Area Under Curve
        }
        
        return metrics


def roc_curve_from_scratch(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Convenience function to compute ROC curve in one step.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores or probabilities
        
    Returns:
        Tuple containing:
        - False positive rates
        - True positive rates
        - Thresholds
        - AUC value
    """
    roc = ROCCurve()
    fpr, tpr, thresholds = roc.compute_roc(y_true, y_scores)
    return fpr, tpr, thresholds, roc.auc


if __name__ == "__main__":
    # Simple example
    # Generate synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    # Generate scores that are correlated with true labels
    y_scores = y_true * 0.8 + np.random.normal(0, 0.5, 100)
    
    # Compute ROC curve 
    roc = ROCCurve()
    fpr, tpr, thresholds = roc.compute_roc(y_true, y_scores)
    
    # Print AUC
    print(f"AUC: {roc.auc:.3f}")
    
    # Find optimal threshold using Youden's J statistic
    optimal_threshold = roc.optimal_threshold(method='youden')
    print(f"Optimal threshold (Youden): {optimal_threshold:.3f}")
    
    # Get metrics at optimal threshold
    metrics = roc.get_performance_metrics(optimal_threshold)
    print(f"Metrics at optimal threshold:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    roc.plot_roc_curve(title="ROC Curve Example")
    plt.tight_layout()
    plt.show()





