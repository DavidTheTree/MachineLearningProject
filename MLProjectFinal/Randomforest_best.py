from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# Optimized Random Forest parameters
best_params_rf = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

# Initialize and train the Random Forest model with optimized parameters
rf_optimized = RandomForestClassifier(**best_params_rf, random_state=42)
rf_optimized.fit(X_train_balanced, y_train_balanced)

# Evaluate on validation set
rf_y_proba_val = rf_optimized.predict_proba(X_val_split)[:, 1]
rf_y_pred_val = rf_optimized.predict(X_val_split)

rf_val_accuracy = accuracy_score(y_val_split, rf_y_pred_val)
rf_val_precision = precision_score(y_val_split, rf_y_pred_val, pos_label=1)
rf_val_recall = recall_score(y_val_split, rf_y_pred_val, pos_label=1)
rf_val_f1 = f1_score(y_val_split, rf_y_pred_val, pos_label=1)
rf_val_auc = roc_auc_score(y_val_split, rf_y_proba_val)

print(f"Validation Metrics:")
print(f"Accuracy: {rf_val_accuracy:.4f}")
print(f"Precision: {rf_val_precision:.4f}")
print(f"Recall: {rf_val_recall:.4f}")
print(f"F1 Score: {rf_val_f1:.4f}")
print(f"AUC: {rf_val_auc:.4f}")

# Evaluate on test set (if labeled)
if labeled_test:
    rf_y_proba_test = rf_optimized.predict_proba(X_test_final)[:, 1]
    rf_y_pred_test = rf_optimized.predict(X_test_final)

    rf_test_accuracy = accuracy_score(y_test, rf_y_pred_test)
    rf_test_precision = precision_score(y_test, rf_y_pred_test, pos_label=1)
    rf_test_recall = recall_score(y_test, rf_y_pred_test, pos_label=1)
    rf_test_f1 = f1_score(y_test, rf_y_pred_test, pos_label=1)
    rf_test_auc = roc_auc_score(y_test, rf_y_proba_test)

    print(f"\nTest Metrics:")
    print(f"Accuracy: {rf_test_accuracy:.4f}")
    print(f"Precision: {rf_test_precision:.4f}")
    print(f"Recall: {rf_test_recall:.4f}")
    print(f"F1 Score: {rf_test_f1:.4f}")
    print(f"AUC: {rf_test_auc:.4f}")

    # Plot AUC Graph for test set
    fpr, tpr, _ = roc_curve(y_test, rf_y_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {rf_test_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guess")
    plt.title("ROC Curve for Random Forest on Test Data")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
else:
    # Predict on test data if labels are unavailable
    rf_y_pred_test = rf_optimized.predict(X_test_final)
    print("\nPredictions on Unlabeled Test Data:")
    print(rf_y_pred_test)