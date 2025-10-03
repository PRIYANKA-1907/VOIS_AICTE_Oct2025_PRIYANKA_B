# 2. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# =============================
# 3. Upload and Load Dataset
# =============================
uploaded = files.upload()
file_path = list(uploaded.keys())[0]   # automatically get uploaded filename
df = pd.read_excel(file_path)

# Preview dataset
print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# =============================
# 4. Data Cleaning
# =============================

# Drop duplicates
df = df.drop_duplicates()

# Define important columns (use only if available in dataset)
required_cols = ["name", "host_name", "neighbourhood", "price"]
available_cols = [col for col in required_cols if col in df.columns]

# Drop missing values in available important columns
if available_cols:
    df = df.dropna(subset=available_cols)

# Handle price column
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")   # convert to numeric
    df = df[df["price"] <= 1000]  # remove extreme outliers

print("\nData after cleaning:", df.shape)

# =============================
# 5. Exploratory Data Analysis (EDA)
# =============================

# --- Average price per neighbourhood group
if "neighbourhood_group" in df.columns and "price" in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="neighbourhood_group", y="price", data=df, estimator=lambda x: sum(x)/len(x))
    plt.title("Average Price by Neighbourhood Group")
    plt.xticks(rotation=45)
    plt.show()

# --- Distribution of room types
if "room_type" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="room_type", data=df, order=df["room_type"].value_counts().index)
    plt.title("Distribution of Room Types")
    plt.show()

# --- Availability analysis
if "availability_365" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["availability_365"], bins=50, kde=False)
    plt.title("Distribution of Availability (days/year)")
    plt.xlabel("Days available per year")
    plt.ylabel("Number of Listings")
    plt.show()

# --- Top 10 neighbourhoods with highest listings
if "neighbourhood" in df.columns:
    plt.figure(figsize=(10,6))
    top_neighbourhoods = df["neighbourhood"].value_counts().head(10)
    sns.barplot(x=top_neighbourhoods.values, y=top_neighbourhoods.index)
    plt.title("Top 10 Neighbourhoods by Listing Count")
    plt.xlabel("Number of Listings")
    plt.show()

# --- Correlation Heatmap (numerical features)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =============================
# 6. Summary Insights
# =============================
if "price" in df.columns:
    print("Average price:", round(df["price"].mean(), 2))

if "room_type" in df.columns:
    print("Most common room type:", df["room_type"].mode()[0])

if "neighbourhood_group" in df.columns:
    print("Neighbourhood group with most listings:", df["neighbourhood_group"].mode()[0])
