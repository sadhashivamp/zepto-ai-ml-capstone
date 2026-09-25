import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

df = sns.load_dataset("titanic")


# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])


# Select features and target
features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked"
]

X = df[features]
y = df["survived"]


# Convert categorical columns into numbers
X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Standardize age and fare
scaler = StandardScaler()

X_train[["age", "fare"]] = scaler.fit_transform(
    X_train[["age", "fare"]]
)

X_test[["age", "fare"]] = scaler.transform(
    X_test[["age", "fare"]]
)


print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nFeatures:")
print(X_train.columns.tolist())


# Baseline classification models
logistic_model = LogisticRegression(
    random_state=42
)

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

random_forest_model = RandomForestClassifier(
    random_state=42
)


logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


print("\nModels trained successfully!")


# Baseline model evaluation
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


print("\n========== BASELINE MODELS ==========")

for name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# Class weight balanced models
balanced_logistic = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

balanced_tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

balanced_forest = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)


balanced_models = {
    "Balanced Logistic Regression": balanced_logistic,
    "Balanced Decision Tree": balanced_tree,
    "Balanced Random Forest": balanced_forest
}


print("\n========== CLASS WEIGHT BALANCED ==========")

for name, model in balanced_models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\n========== SMOTE ==========")

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


smote_forest = RandomForestClassifier(
    random_state=42
)

smote_forest.fit(
    X_train_smote,
    y_train_smote
)

smote_predictions = smote_forest.predict(X_test)
smote_probabilities = smote_forest.predict_proba(X_test)[:, 1]


print("\nSMOTE Random Forest")
print("Accuracy:", accuracy_score(y_test, smote_predictions))
print("Precision:", precision_score(y_test, smote_predictions))
print("Recall:", recall_score(y_test, smote_predictions))
print("F1 Score:", f1_score(y_test, smote_predictions))
print("ROC-AUC:", roc_auc_score(y_test, smote_probabilities))

print("Confusion Matrix:")
print(confusion_matrix(y_test, smote_predictions))

# Random Forest GridSearchCV

random_forest = RandomForestClassifier(
    random_state=42,
    oob_score=True
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    random_forest,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n========== RANDOM FOREST GRID SEARCH ==========")

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(grid_search.best_score_)

print("\nOOB score:")
print(best_model.oob_score_)

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nTest set results:")
print("Accuracy:", accuracy_score(y_test, best_predictions))
print("Precision:", precision_score(y_test, best_predictions))
print("Recall:", recall_score(y_test, best_predictions))
print("F1 Score:", f1_score(y_test, best_predictions))
print("ROC-AUC:", roc_auc_score(y_test, best_probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))