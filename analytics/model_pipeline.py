import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


df = sns.load_dataset("titanic")


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


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features="log2",
    random_state=42
)


pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline.fit(X_train, y_train)


print("Pipeline trained successfully!")


model_path = "analytics/titanic_model.joblib"

joblib.dump(
    pipeline,
    model_path
)

print("Pipeline saved to:", model_path)


# Reload pipeline
loaded_pipeline = joblib.load(model_path)

print("Pipeline loaded successfully!")


# Raw input
raw_input = pd.DataFrame({
    "pclass": [1],
    "age": [30],
    "sibsp": [0],
    "parch": [0],
    "fare": [80],
    "sex": ["female"],
    "embarked": ["S"]
})


prediction = loaded_pipeline.predict(raw_input)
probability = loaded_pipeline.predict_proba(raw_input)[0][1]


print("\nRaw input:")
print(raw_input)

print("\nPrediction:", prediction[0])
print("Survival probability:", probability)