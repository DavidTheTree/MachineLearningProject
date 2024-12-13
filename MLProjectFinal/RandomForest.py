from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load datasets
train_df = pd.read_csv('./MLProject/train.csv')

# Preprocessing
X_train = train_df.drop(columns=['income>50K'])
y_train = train_df['income>50K']

numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Handle missing values and preprocess
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])

X_train_final = np.hstack((X_train_scaled, X_train_encoded))

# Split data and apply SMOTE
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform Grid Search
rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train_balanced, y_train_balanced)

# Best parameters and score
print("\nBest Parameters for Random Forest:", rf_grid_search.best_params_)
print("Best AUC for Random Forest:", rf_grid_search.best_score_)

# Train optimized model
rf_optimized = rf_grid_search.best_estimator_
rf_optimized.fit(X_train_balanced, y_train_balanced)
rf_y_proba = rf_optimized.predict_proba(X_val_split)[:, 1]

# Metrics
rf_auc = roc_auc_score(y_val_split, rf_y_proba)
fpr, tpr, _ = roc_curve(y_val_split, rf_y_proba)

print(f"Random Forest - Optimized AUC: {rf_auc:.4f}")

# Plot AUC Graph
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guess")
plt.title("ROC Curve for Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
