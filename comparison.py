import matplotlib.pyplot as plt

# Example scores (replace with your actual results)
log_reg_f1 = 0.55
svm_f1 = 0.55

log_reg_precision = 0.45
svm_precision = 0.45

log_reg_recall = 0.71
svm_recall = 0.73

# Collect metrics
metrics = ['F1 Score', 'Precision', 'Recall']
log_reg_scores = [log_reg_f1, log_reg_precision, log_reg_recall]
svm_scores = [svm_f1, svm_precision, svm_recall]

# Plot horizontal bars
bar_width = 0.35
index = range(len(metrics))

plt.figure(figsize=(8,6))
plt.barh(index, log_reg_scores, bar_width, label='Logistic Regression', color='skyblue')
plt.barh([i + bar_width for i in index], svm_scores, bar_width, label='SVM', color='salmon')

plt.yticks([i + bar_width/2 for i in index], metrics)
plt.xlabel('Score')
plt.title('Model Performance Comparison (Progress Bar Style)')
plt.legend()
plt.xlim(0,1)  # since scores are between 0 and 1
plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')
# plt.show()
