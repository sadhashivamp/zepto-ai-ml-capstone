import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("analytics/titanic.csv")

df = df.dropna(subset=["embarked", "embark_town"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.drop(columns=["deck"])

print("Before standardization:")
print("Age mean:", df["age"].mean())
print("Age std:", df["age"].std())
print("Fare mean:", df["fare"].mean())
print("Fare std:", df["fare"].std())

scaler = StandardScaler()

df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

print("\nAfter standardization:")
print("Age mean:", df["age"].mean())
print("Age std:", df["age"].std())
print("Fare mean:", df["fare"].mean())
print("Fare std:", df["fare"].std())