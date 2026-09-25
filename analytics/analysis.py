import pandas as pd
import seaborn as sns


df = sns.load_dataset("titanic")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

df = df.drop(
    columns=["deck", "embark_town"]
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())