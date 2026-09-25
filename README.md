# Zepto AI/ML Capstone

## Module 1 - Data Pipeline

This module scrapes book data from Books to Scrape, cleans the data, stores it in SQLite, and runs SQL queries.

## Installation

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

Install the required packages:

pip install -r requirements.txt
Run

Run the scraper:

python data_pipeline/scraper/book_scraper.py

Run the cleaning script:

python data_pipeline/cleaning/clean_books.py

Create the database:

python data_pipeline/database/load_database.py

Run the SQL queries:

python data_pipeline/queries/queries.py
Data Cleaning
Price was converted from text to numeric GBP.
Ratings were converted from One to Five into numbers 1 to 5.
Availability was converted into a boolean in_stock column.
Missing values were checked.
Duplicate rows were checked.

No missing values or duplicate rows were present in the final cleaned data.

Currency Conversion

The fixed conversion rate used is:

1 GBP = 105.50 INR

The INR price is calculated using:

price_inr = price_gbp * 105.50
Database

SQLite is used to store the cleaned data.

The database contains two related tables:

categories
books

The books table uses category_id as a foreign key.

SQL Queries

Five SQL queries were implemented covering:

SELECT and WHERE
ORDER BY and LIMIT
DISTINCT
BETWEEN
JOIN

pandas.read_sql() is used for the query results.

The SQL JOIN result is also reproduced using Pandas merge() and compared with the SQL result.


