import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

missing_percentage = df.isnull().mean() * 100

print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing percentage:")
print(missing_percentage[missing_percentage > 0])

df.to_csv("analytics/titanic.csv", index=False)

print("\nDataset saved successfully!")
print("analytics/titanic.csv")

# Missing value handling
df = df.dropna(subset=["embarked", "embark_town"])

df["age"] = df["age"].fillna(df["age"].median())

df = df.drop(columns=["deck"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal shape:", df.shape)

print("\nCleaning decisions:")
print("age: 19.8653% missing -> median imputation")
print("embarked: 0.2245% missing -> rows dropped")
print("deck: 77.2166% missing -> column dropped")
print("embark_town: 0.2245% missing -> rows dropped")