import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = sns.load_dataset("titanic")


# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

df = df.drop(
    columns=["deck", "embark_town"]
)


sns.set_theme(style="whitegrid")


# 1. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()


# 2. Fare distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# 3. Survival by sex
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="sex",
    y="survived"
)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# 4. Survival by passenger class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# 5. Survival by sex and passenger class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# 6. Correlation heatmap
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation = df[correlation_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()