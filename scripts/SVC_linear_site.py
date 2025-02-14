import pandas as pd
import numpy as np
import os
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


# import data
X = np.genfromtxt('.../SVC_data.csv', delimiter=",")
y = np.genfromtxt('.../Classification_site_labels.csv')


# Define the number of folds for nested cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Outer loop for evaluation
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Inner loop for hyperparameter tuning

# Define hyperparameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]  # Regularization strength for LinearSVC
}

# Lists to store results
outer_accuracies = []
outer_balanced_accuracies = []
best_params_list = []
outer_sensitivities = []
outer_specificities = []

# Outer loop: Model evaluation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner loop: Hyperparameter tuning
    grid_search = GridSearchCV(LinearSVC(dual=False, random_state=42, max_iter=1000, class_weight='balanced'), 
                               param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model evaluation on the outer test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Compute accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

   # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Compute sensitivity (recall) and specificity for each class
    sensitivity = []
    specificity = []
    for i in range(len(np.unique(y))):  # Loop through each class
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)
        
        sensitivity.append(TP / (TP + FN))  # Sensitivity (Recall)
        specificity.append(TN / (TN + FP))  # Specificity



    outer_accuracies.append(accuracy)
    outer_balanced_accuracies.append(balanced_accuracy)
    outer_sensitivities.append(sensitivity)
    outer_specificities.append(specificity)
    best_params_list.append(grid_search.best_params_)

    #print(f'Best parameters: {grid_search.best_params_}')
    #print(f'Outer fold test accuracy: {accuracy:.4f}')
    #print(f'Outer fold balanced accuracy: {balanced_accuracy:.4f}')

# Compute final statistics
final_stats = {
    "mean_accuracy": np.mean(outer_accuracies),
    "std_accuracy": np.std(outer_accuracies),
    "mean_balanced_accuracy": np.mean(outer_balanced_accuracies),
    "std_balanced_accuracy": np.std(outer_balanced_accuracies),
    "mean_sensitivity": np.mean(outer_sensitivities),  # Mean sensitivity per class
    "std_sensitivity": np.std(outer_sensitivities),    # Std sensitivity per class
    "mean_specificity": np.mean(outer_specificities),  # Mean specificity per class
    "std_specificity": np.std(outer_specificities),    # Std specificity per class
    "best_params_per_fold": best_params_list
}


# Print final performance statistics
# print(f'\nFinal Nested CV Accuracy: {final_stats["mean_accuracy"]:.4f} ± {final_stats["std_accuracy"]:.4f}')
# print(f'Final Nested CV Balanced Accuracy: {final_stats["mean_balanced_accuracy"]:.4f} ± {final_stats["std_balanced_accuracy"]:.4f}')

# Save results to a JSON file
with open(".../nested_CV_results_SVC.json", "w") as file:
    json.dump(final_stats, file, indent=4, ensure_ascii=False)

# print("Results saved to nested_cv_results.json")
