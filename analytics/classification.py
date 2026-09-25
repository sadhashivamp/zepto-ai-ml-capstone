import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE


# Load data
df = pd.read_csv("analytics/titanic.csv")


# Handle missing values
df = df.dropna(subset=["embarked", "embark_town"])

df["age"] = df["age"].fillna(df["age"].median())

df = df.drop(columns=["deck"])


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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())


# Convert categorical columns into numbers
X_train = pd.get_dummies(
    X_train,
    columns=["sex", "embarked"],
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=["sex", "embarked"],
    drop_first=True
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# Fill missing values using training data
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# Standardize age and fare
scaler = StandardScaler()

X_train[["age", "fare"]] = scaler.fit_transform(
    X_train[["age", "fare"]]
)

X_test[["age", "fare"]] = scaler.transform(
    X_test[["age", "fare"]]
)


print("\nPreprocessing completed.")

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


results = []


print("\n========== BASELINE MODELS ==========")


for name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ROC curves
plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=name + " AUC = " + str(round(roc_auc, 3))
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Baseline Models")
plt.legend()
plt.tight_layout()
plt.show()


# Decision Tree visualization
plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree_model,
    feature_names=X_train.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree")
plt.tight_layout()
plt.show()


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

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

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


smote_accuracy = accuracy_score(
    y_test,
    smote_predictions
)

smote_precision = precision_score(
    y_test,
    smote_predictions
)

smote_recall = recall_score(
    y_test,
    smote_predictions
)

smote_f1 = f1_score(
    y_test,
    smote_predictions
)

smote_auc = roc_auc_score(
    y_test,
    smote_probabilities
)


print("\nSMOTE Random Forest")
print("Accuracy:", smote_accuracy)
print("Precision:", smote_precision)
print("Recall:", smote_recall)
print("F1 Score:", smote_f1)
print("ROC-AUC:", smote_auc)

print("Confusion Matrix:")
print(confusion_matrix(y_test, smote_predictions))


results.append({
    "Model": "SMOTE Random Forest",
    "Accuracy": smote_accuracy,
    "Precision": smote_precision,
    "Recall": smote_recall,
    "F1": smote_f1,
    "ROC-AUC": smote_auc
})


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


grid_search.fit(
    X_train,
    y_train
)


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


best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

best_precision = precision_score(
    y_test,
    best_predictions
)

best_recall = recall_score(
    y_test,
    best_predictions
)

best_f1 = f1_score(
    y_test,
    best_predictions
)

best_auc = roc_auc_score(
    y_test,
    best_probabilities
)


print("\nTest set results:")
print("Accuracy:", best_accuracy)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1 Score:", best_f1)
print("ROC-AUC:", best_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))


results.append({
    "Model": "Tuned Random Forest",
    "Accuracy": best_accuracy,
    "Precision": best_precision,
    "Recall": best_recall,
    "F1": best_f1,
    "ROC-AUC": best_auc
})


# Final comparison table
results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df.round(3).to_string(index=False)
)