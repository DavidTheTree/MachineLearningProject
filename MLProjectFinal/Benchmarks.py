import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


# Load preprocessed SMOTEd data
train_df =pd.read_csv('./MLProject/train.csv', header =0)
test_df = pd.read_csv('./MLProject/test.csv', header =0)

# Separate features and target
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

# Define classifiers for benchmarks
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500),
    "Ensemble (Voting)": VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
}

# Evaluate classifiers
results = []
auc_scores = {}

for name, clf in tqdm(classifiers.items(), desc="Model Training and Evaluation", unit="model"):
    clf.fit(X_train_balanced, y_train_balanced)
    y_pred = clf.predict(X_val_split)

    # Calculate metrics
    accuracy = accuracy_score(y_val_split, y_pred)
    precision = precision_score(y_val_split, y_pred, pos_label=1)
    recall = recall_score(y_val_split, y_pred, pos_label=1)
    f1 = f1_score(y_val_split, y_pred, pos_label=1)

    # Calculate AUC
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_val_split)[:, 1]
    else:
        y_proba = clf.decision_function(X_val_split)
    auc = roc_auc_score(y_val_split, y_proba)
    auc_scores[name] = auc

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the results
print("\nFinal Benchmark Results:")
print(results_df)

# Save results
results_df.to_csv("benchmark_results_with_auc.csv", index=False)

# Plot the ROC curve
plt.figure(figsize=(10, 8))
for name, clf in classifiers.items():
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_val_split)[:, 1]
    else:
        y_proba = clf.decision_function(X_val_split)

    fpr, tpr, _ = roc_curve(y_val_split, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_scores[name]:.2f})")

# Plot formatting
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guess")
plt.title("ROC Curves for Benchmark Models", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()