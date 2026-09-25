import pandas as pd


input_path = "data_pipeline/data/raw_books.csv"
output_path = "data_pipeline/data/clean_books.csv"


df = pd.read_csv(input_path)

print("Data loaded successfully!")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())


# Clean price
df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .str.replace("Â", "", regex=False)
)

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

if df["price_gbp"].isnull().any():
    df["price_gbp"] = df["price_gbp"].fillna(
        df["price_gbp"].median()
    )

df["price_inr"] = df["price_gbp"] * 105.50


# Convert star rating
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

if df["rating"].isnull().any():
    df["rating"] = df["rating"].fillna(
        df["rating"].median()
    )


# Convert availability to boolean
df["in_stock"] = (
    df["availability"]
    .str.contains("In stock", case=False, na=False)
)


# Remove old columns
df = df.drop(
    columns=["price", "star_rating", "availability"]
)


print("\nCleaned data:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())


df.to_csv(output_path, index=False)

print("\nClean data saved successfully!")
print(output_path)