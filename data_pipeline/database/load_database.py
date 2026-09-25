import sqlite3
import pandas as pd


CSV_PATH = "data_pipeline/data/clean_books.csv"
DB_PATH = "data_pipeline/books.db"


df = pd.read_csv(CSV_PATH)

connection = sqlite3.connect(DB_PATH)

cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


for category in df["category"].unique():

    cursor.execute(
        """
        INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


for _, row in df.iterrows():

    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )


connection.commit()


print("Database created successfully!")

print("\nCategories:")
for row in cursor.execute("SELECT * FROM categories"):
    print(row)

print("\nTotal books:")
cursor.execute("SELECT COUNT(*) FROM books")
print(cursor.fetchone()[0])

connection.close()