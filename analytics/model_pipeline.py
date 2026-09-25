import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


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


# Feature types
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


# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])


# Preprocessor
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


# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features="log2",
    random_state=42
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train pipeline
pipeline.fit(
    X_train,
    y_train
)


print("Pipeline trained successfully!")


# Save pipeline
model_path = "analytics/titanic_model.joblib"

joblib.dump(
    pipeline,
    model_path
)

print("Pipeline saved to:", model_path)


# Load pipeline
loaded_pipeline = joblib.load(
    model_path
)

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


# Prediction
prediction = loaded_pipeline.predict(
    raw_input
)

probability = loaded_pipeline.predict_proba(
    raw_input
)[0][1]


print("\nRaw input:")
print(raw_input)

print("\nPrediction:", prediction[0])
print("Survival probability:", probability)