import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('./MLProject/train.csv')
test_df = pd.read_csv('./MLProject/test.csv')

# Separate features and target for training data
X_train = train_df.drop(columns=['income>50K'])
y_train = train_df['income>50K']

# Check if the test dataset includes labels
if 'income>50K' in test_df.columns:
    y_test = test_df['income>50K']  # Extract test labels
    X_test = test_df.drop(columns=['ID', 'income>50K'])  # Drop ID and target column
    labeled_test = True
else:
    X_test = test_df.drop(columns=['ID'])  # Drop ID column
    labeled_test = False

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = scaler.transform(X_test[numerical_cols])

# Combine numerical and encoded categorical features
X_train_final = np.hstack((X_train_scaled, X_train_encoded))
X_test_final = np.hstack((X_test_scaled, X_test_encoded))

# Split training data and apply SMOTE
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

# Define parameter grid for Gradient Boosting
gb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform Grid Search for optimization
gb = GradientBoostingClassifier(random_state=42)
gb_grid_search = GridSearchCV(estimator=gb, param_grid=gb_param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)
gb_grid_search.fit(X_train_balanced, y_train_balanced)

# Best parameters and score
print("\nBest Parameters for Gradient Boosting:", gb_grid_search.best_params_)
print("Best AUC for Gradient Boosting:", gb_grid_search.best_score_)

# Train optimized model
gb_optimized = gb_grid_search.best_estimator_
gb_optimized.fit(X_train_balanced, y_train_balanced)

# Evaluate on validation set
gb_y_proba_val = gb_optimized.predict_proba(X_val_split)[:, 1]
gb_y_pred_val = gb_optimized.predict(X_val_split)

gb_val_accuracy = accuracy_score(y_val_split, gb_y_pred_val)
gb_val_precision = precision_score(y_val_split, gb_y_pred_val, pos_label=1)
gb_val_recall = recall_score(y_val_split, gb_y_pred_val, pos_label=1)
gb_val_f1 = f1_score(y_val_split, gb_y_pred_val, pos_label=1)
gb_val_auc = roc_auc_score(y_val_split, gb_y_proba_val)

print(f"\nValidation Metrics:")
print(f"Accuracy: {gb_val_accuracy:.4f}")
print(f"Precision: {gb_val_precision:.4f}")
print(f"Recall: {gb_val_recall:.4f}")
print(f"F1 Score: {gb_val_f1:.4f}")
print(f"AUC: {gb_val_auc:.4f}")

# Evaluate on test set (if labeled)
if labeled_test:
    gb_y_proba_test = gb_optimized.predict_proba(X_test_final)[:, 1]
    gb_y_pred_test = gb_optimized.predict(X_test_final)

    gb_test_accuracy = accuracy_score(y_test, gb_y_pred_test)
    gb_test_precision = precision_score(y_test, gb_y_pred_test, pos_label=1)
    gb_test_recall = recall_score(y_test, gb_y_pred_test, pos_label=1)
    gb_test_f1 = f1_score(y_test, gb_y_pred_test, pos_label=1)
    gb_test_auc = roc_auc_score(y_test, gb_y_proba_test)

    print(f"\nTest Metrics:")
    print(f"Accuracy: {gb_test_accuracy:.4f}")
    print(f"Precision: {gb_test_precision:.4f}")
    print(f"Recall: {gb_test_recall:.4f}")
    print(f"F1 Score: {gb_test_f1:.4f}")
    print(f"AUC: {gb_test_auc:.4f}")

    # Plot AUC Graph for test set
    fpr, tpr, _ = roc_curve(y_test, gb_y_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Gradient Boosting (AUC = {gb_test_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guess")
    plt.title("ROC Curve for Gradient Boosting on Test Data")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
else:
    # Predict on test data if labels are unavailable
    gb_y_pred_test = gb_optimized.predict(X_test_final)
    print("\nPredictions on Unlabeled Test Data:")
    print(gb_y_pred_test)
