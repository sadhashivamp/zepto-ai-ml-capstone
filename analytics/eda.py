import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("analytics/titanic.csv")

df = df.dropna(subset=["embarked", "embark_town"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.drop(columns=["deck"])

sns.set_theme(style="whitegrid")

# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Age boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Fare boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()
plt.show()


# IQR outliers
def count_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((df[column] < lower) | (df[column] > upper)).sum()


print("\nIQR Outliers:")
print("Age:", count_outliers("age"))
print("Fare:", count_outliers("fare"))


# Fare statistics
fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\nFare statistics:")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

print("\nFare skewness:")

if fare_mean > fare_median > fare_mode:
    print("Fare is right-skewed because mean > median > mode.")
elif fare_mean < fare_median < fare_mode:
    print("Fare is left-skewed because mean < median < mode.")
else:
    print("Fare does not clearly follow the simple mean-median-mode pattern.")


# Survival by sex using boolean masking
female_survival = df[df["sex"] == "female"]["survived"].mean()
male_survival = df[df["sex"] == "male"]["survived"].mean()

print("\nSurvival rate by sex:")
print("Female:", female_survival)
print("Male:", male_survival)


# Survival by passenger class using boolean masking
first_class_survival = df[df["pclass"] == 1]["survived"].mean()
second_class_survival = df[df["pclass"] == 2]["survived"].mean()
third_class_survival = df[df["pclass"] == 3]["survived"].mean()

print("\nSurvival rate by passenger class:")
print("1st class:", first_class_survival)
print("2nd class:", second_class_survival)
print("3rd class:", third_class_survival)


# Survival by sex and passenger class using boolean masking
print("\nSurvival rate by sex and passenger class:")

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        survival_rate = df[
            (df["sex"] == sex) & (df["pclass"] == pclass)
        ]["survived"].mean()

        print(sex, "class", pclass, ":", survival_rate)


# Survival by sex
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# Survival by passenger class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# Survival by sex and passenger class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# Correlation heatmap
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation = df[correlation_columns].corr()

print("\nCorrelation matrix:")
print(correlation)

print("\nStrongest correlations with survived:")

survived_correlation = correlation["survived"].drop("survived")
print(survived_correlation.abs().sort_values(ascending=False).head(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()