import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


# Load data
df = pd.read_csv("analytics/titanic.csv")


# Handle missing values
df = df.dropna(subset=["embarked", "embark_town"])

df["age"] = df["age"].fillna(df["age"].median())

df = df.drop(columns=["deck"])


# Select features and target
features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X = df[features]
y = df["fare"]


# Convert categorical columns into numbers
X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)


# Convert boolean columns into numbers
X = X.astype(float)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Fill missing values using training data
imputer = SimpleImputer(
    strategy="median"
)

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


# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# Train linear regression model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Predictions
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# Adjusted R2
n = len(y_test)
p = X_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)


print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nRegression Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)


# Residuals
residuals = y_test - predictions


# Residual plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=predictions,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()


# Breusch-Pagan test
X_test_with_constant = sm.add_constant(
    X_test
)

bp_test = het_breuschpagan(
    residuals,
    X_test_with_constant
)

lm_statistic = bp_test[0]
lm_pvalue = bp_test[1]


print("\nBreusch-Pagan Test")
print("LM Statistic:", lm_statistic)
print("LM p-value:", lm_pvalue)


if lm_pvalue < 0.05:
    print("Evidence of heteroscedasticity detected.")
else:
    print("No significant evidence of heteroscedasticity detected.")