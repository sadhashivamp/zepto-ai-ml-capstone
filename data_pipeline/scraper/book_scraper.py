import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
MIN_CATEGORIES = 3
BOOKS_PER_CATEGORY = 20
TARGET_BOOKS = 60


def get_soup(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")

def get_categories():
    soup = get_soup(BASE_URL)

    categories = {}

    links = soup.select("div.side_categories ul li ul li a")

    for link in links:
        name = link.get_text(strip=True)
        url = urljoin(BASE_URL, link.get("href"))
        categories[name] = url

    return categories


def get_rating(book):
    rating = book.select_one("p.star-rating")

    if rating:
        classes = rating.get("class", [])

        for value in ["One", "Two", "Three", "Four", "Five"]:
            if value in classes:
                return value

    return None


def scrape_category(category_name, category_url, limit):

    books_data = []
    current_url = category_url

    while current_url and len(books_data) < limit:

        print(f"Scraping {category_name}: {current_url}")

        soup = get_soup(current_url)

        books = soup.select("article.product_pod")

        for book in books:

            title = book.h3.a.get("title", "").strip()

            price = book.select_one(".price_color")
            price = price.get_text(strip=True) if price else None

            rating = get_rating(book)

            availability = book.select_one(".availability")
            availability = (
                availability.get_text(" ", strip=True)
                if availability
                else None
            )

            books_data.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category_name
            })

            if len(books_data) >= limit:
                break

        next_link = soup.select_one("li.next a")

        if next_link:
            current_url = urljoin(
                current_url,
                next_link.get("href")
            )
        else:
            current_url = None

    return books_data


def main():

    print("Starting book scraper...\n")

    categories = get_categories()

    print("Total categories available:", len(categories))

    all_books = []
    selected_categories = []

    for category_name, category_url in categories.items():

        if (
            len(all_books) >= TARGET_BOOKS
            and len(selected_categories) >= MIN_CATEGORIES
        ):
            break

        print("\nScraping category:", category_name)

        books = scrape_category(
            category_name,
            category_url,
            BOOKS_PER_CATEGORY
        )

        all_books.extend(books)
        selected_categories.append(category_name)

        print(
            category_name,
            "->",
            len(books),
            "books"
        )

        print(
            "Total collected so far:",
            len(all_books)
        )

    df = pd.DataFrame(all_books)

    print("\n========== SCRAPING SUMMARY ==========")

    print("Total books:", len(df))

    print("\nCategories:")
    print(df["category"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    output_path = "data_pipeline/data/raw_books.csv"

    df.to_csv(output_path, index=False)

    print("\nRaw data saved to:", output_path)


if __name__ == "__main__":
    main()