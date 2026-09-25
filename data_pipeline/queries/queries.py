import sqlite3
import pandas as pd


DB_PATH = "data_pipeline/books.db"

connection = sqlite3.connect(DB_PATH)


# Query 1 - SELECT and WHERE

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
"""

print("\n========== QUERY 1 ==========")
print(query1)

result1 = pd.read_sql(query1, connection)
print(result1)


# Query 2 - ORDER BY and LIMIT

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

print("\n========== QUERY 2 ==========")
print(query2)

result2 = pd.read_sql(query2, connection)
print(result2)


# Query 3 - DISTINCT

query3 = """
SELECT DISTINCT rating
FROM books
ORDER BY rating
"""

print("\n========== QUERY 3 ==========")
print(query3)

result3 = pd.read_sql(query3, connection)
print(result3)


# Query 4 - BETWEEN

query4 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp
"""

print("\n========== QUERY 4 ==========")
print(query4)

result4 = pd.read_sql(query4, connection)
print(result4)


# Query 5 - JOIN

query5 = """
SELECT
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.title ASC
LIMIT 10
"""

print("\n========== QUERY 5 ==========")
print(query5)

result5 = pd.read_sql(query5, connection)
print(result5)


# Pandas merge

books_df = pd.read_sql(
    "SELECT * FROM books",
    connection
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id"
)

merged_result = (
    merged_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "category_name"
        ]
    ]
    .sort_values(
        ["rating", "title"],
        ascending=[False, True]
    )
    .head(10)
    .reset_index(drop=True)
)

print("\n========== PANDAS MERGE ==========")
print(merged_result)


# Compare SQL JOIN and Pandas merge

sql_result = (
    result5
    .sort_values(
        ["rating", "title"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

print("\n========== JOIN COMPARISON ==========")
print("SQL JOIN and Pandas merge match:", sql_result.equals(merged_result))


connection.close()