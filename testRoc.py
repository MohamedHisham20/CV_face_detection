import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score

# Import your ROCCurve implementation
from ROC import ROCCurve  # Replace with actual module name

# 1. Generate synthetic data
np.random.seed(0)
n_samples = 10_000  # You can increase for stress testing
y_true = np.random.randint(0, 2, size=n_samples)
y_scores = y_true * 0.9 + np.random.normal(0, 0.4, size=n_samples)

# 2. Compute using your implementation
roc_custom = ROCCurve()
start_custom = time.time()
fpr_custom, tpr_custom, _ = roc_custom.compute_roc(y_true, y_scores)
auc_custom = roc_custom.auc
time_custom = time.time() - start_custom

# 3. Compute using sklearn
start_sk = time.time()
fpr_sk, tpr_sk, _ = sk_roc_curve(y_true, y_scores)
auc_sk = roc_auc_score(y_true, y_scores)
time_sk = time.time() - start_sk

# 4. Print results
print("=== Performance Comparison ===")
print(f"Custom ROC AUC:     {auc_custom:.6f}")
print(f"Sklearn ROC AUC:    {auc_sk:.6f}")
print(f"Difference in AUC:  {abs(auc_custom - auc_sk):.6f}")
print(f"Custom ROC Time:    {time_custom:.6f} seconds")
print(f"Sklearn ROC Time:   {time_sk:.6f} seconds")

# 5. Plotting
plt.figure(figsize=(10, 6))
plt.plot(fpr_custom, tpr_custom, label=f'Custom ROC (AUC = {auc_custom:.3f})', lw=2)
plt.plot(fpr_sk, tpr_sk, label=f'sklearn ROC (AUC = {auc_sk:.3f})', linestyle='--', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Custom vs. Sklearn")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
