# Zepto AI/ML Capstone

This project contains the work for the AI/ML capstone assessment.

## Module 1 - Data Pipeline

In Module 1, I scraped book data from Books to Scrape, cleaned the data, stored it in SQLite and ran SQL queries.

### Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

### Scraping

Run:

```bash
python data_pipeline/scraper/book_scraper.py
```

The scraper collects:

- title
- price
- rating
- availability
- category

The final data contains 71 books from 4 categories.

Raw data is saved in:

```text
data_pipeline/data/raw_books.csv
```

### Data Cleaning

Run:

```bash
python data_pipeline/cleaning/clean_books.py
```

I converted:

- price to numeric GBP
- rating from words to numbers 1 to 5
- availability to True/False

I also checked missing values and duplicate rows.

For parsing failures, median values were used for numeric columns.

The final cleaned data did not have missing values or duplicate rows.

Clean data is saved in:

```text
data_pipeline/data/clean_books.csv
```

### Currency Conversion

I used the fixed conversion rate:

```text
1 GBP = 105.50 INR
```

The INR price is calculated as:

```text
price_inr = price_gbp * 105.50
```

### SQLite Database

Run:

```bash
python data_pipeline/database/load_database.py
```

The database is:

```text
data_pipeline/books.db
```

It contains two tables:

- categories
- books

The `books` table uses `category_id` as the foreign key.

### SQL Queries

Run:

```bash
python data_pipeline/queries/queries.py
```

I used queries for:

- SELECT and WHERE
- ORDER BY and LIMIT
- DISTINCT
- BETWEEN
- JOIN

I also used `pd.read_sql()` for the query results.

The SQL JOIN result was compared with Pandas `merge()` and both results matched.

---

## Module 2 - Titanic Analysis

In Module 2, I used the Titanic dataset from Seaborn and worked on data cleaning, EDA, classification, regression and model pipeline.

### Dataset Analysis

Run:

```bash
python analytics/analysis.py
```

The Titanic dataset has 891 rows and 15 columns.

Missing values were handled as follows:

- `age` - median value
- `embarked` - mode value
- `deck` - dropped because of many missing values
- `embark_town` - dropped

After cleaning, there were no missing values in the remaining columns.

### EDA

Run:

```bash
python analytics/eda.py
```

I created the following charts:

- Age distribution
- Fare distribution
- Survival by sex
- Survival by passenger class
- Survival by sex and passenger class
- Correlation heatmap

From the charts, I observed that female passengers had higher survival rates than male passengers and first-class passengers had higher survival rates than third-class passengers.

The fare distribution was also right-skewed.

### Classification

Run:

```bash
python analytics/classification.py
```

I used:

- Logistic Regression
- Decision Tree
- Random Forest

I checked:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Baseline Results

Logistic Regression:

```text
Accuracy: 0.8045
Precision: 0.7931
Recall: 0.6667
F1 Score: 0.7244
ROC-AUC: 0.8445
```

Decision Tree:

```text
Accuracy: 0.8212
Precision: 0.7937
Recall: 0.7246
F1 Score: 0.7576
ROC-AUC: 0.7949
```

Random Forest:

```text
Accuracy: 0.8156
Precision: 0.7813
Recall: 0.7246
F1 Score: 0.7519
ROC-AUC: 0.8325
```

### Class Imbalance

I also tested class balancing using:

```text
class_weight="balanced"
```

I tested balanced versions of Logistic Regression, Decision Tree and Random Forest.

I also used SMOTE only on the training data.

Before SMOTE:

```text
0    439
1    273
```

After SMOTE:

```text
0    439
1    439
```

SMOTE Random Forest results:

```text
Accuracy: 0.7654
Precision: 0.6800
Recall: 0.7391
F1 Score: 0.7083
ROC-AUC: 0.8267
```

### Random Forest Grid Search

I used GridSearchCV to tune:

- `n_estimators`
- `max_depth`
- `max_features`

Best parameters:

```text
n_estimators = 200
max_depth = 10
max_features = log2
```

Best CV F1 score:

```text
0.7529
```

OOB score:

```text
0.8301
```

Test results:

```text
Accuracy: 0.8156
Precision: 0.8103
Recall: 0.6812
F1 Score: 0.7402
ROC-AUC: 0.8383
```

### Regression

Run:

```bash
python analytics/regression.py
```

I used Linear Regression to predict `fare`.

Results:

```text
MAE: 20.90
RMSE: 30.53
R2: 0.398
Adjusted R2: 0.369
```

I also created a residual plot.

I used the Breusch-Pagan test to check heteroscedasticity.

Results:

```text
LM Statistic: 25.5844
LM p-value: 0.00124
```

The result shows evidence of heteroscedasticity.

### Model Pipeline

Run:

```bash
python analytics/model_pipeline.py
```

I created a complete preprocessing and Random Forest pipeline.

The pipeline includes:

- missing value handling
- scaling
- one-hot encoding
- Random Forest model

The trained pipeline is saved as:

```text
analytics/titanic_model.joblib
```

I also loaded the saved model again and tested it with raw input.

Example:

```text
pclass: 1
age: 30
sibsp: 0
parch: 0
fare: 80
sex: female
embarked: S
```

Prediction:

```text
1
```

Survival probability:

```text
0.9973684210526315
```

## Run Module 2

```bash
python analytics/analysis.py
python analytics/eda.py
python analytics/classification.py
python analytics/regression.py
python analytics/model_pipeline.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Requests
- BeautifulSoup
- Matplotlib
- Seaborn
- SQLite
- Scikit-learn
- Imbalanced-learn
- Statsmodels
- Joblib
- Git