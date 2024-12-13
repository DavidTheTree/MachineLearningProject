import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the datasets
train_df = pd.read_csv('./MLProject/train.csv', header =0)
test_df = pd.read_csv('./MLProject/test.csv', header =0)

# Separate features and target from the training data
X_train = train_df.drop(columns=['income>50K'])
y_train = train_df['income>50K']
X_test = test_df.drop(columns=['ID'])

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to strings for consistent handling
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
X_test[categorical_cols] = X_test[categorical_cols].astype(str)

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

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Apply SMOTE to the training split
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

# Check the distribution of classes after SMOTE
print("Class distribution before SMOTE:")
print(y_train_split.value_counts())
print("Class distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Train a classifier (e.g., Decision Tree)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_balanced, y_train_balanced)

# Evaluate on the validation set
val_score = clf.score(X_val_split, y_val_split)
print(f"Validation Accuracy: {val_score}")

# Optionally, evaluate using cross-validation
cv_scores = cross_val_score(clf, X_train_final, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores)}")

# Convert the SMOTEd data to a DataFrame
X_train_balanced_df = pd.DataFrame(X_train_balanced)
y_train_balanced_df = pd.DataFrame(y_train_balanced, columns=['income>50K'])

# Save to CSV
X_train_balanced_df.to_csv('X_train_smoted.csv', index=False)
y_train_balanced_df.to_csv('y_train_smoted.csv', index=False)

print("SMOTEd datasets saved as CSV files.")