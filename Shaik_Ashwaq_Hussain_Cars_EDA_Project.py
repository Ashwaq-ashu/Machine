# ============================================
# EXPLORATORY DATA ANALYSIS - USED CARS
# ============================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("used_car_dataset.csv")

# ============================================
# 2. DATASET UNDERSTANDING
# ============================================

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types and non-null counts
print("\nDataset Information:")
df.info()

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Basic descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# ============================================
# 3. DATA QUALITY CHECK
# ============================================

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check unique values in categorical columns
print("\nTransmission Values:")
print(df["Transmission"].unique())

print("\nOwner Values:")
print(df["Owner"].unique())

print("\nFuel Type Values:")
print(df["FuelType"].unique())

# ============================================
# 4. INSPECT DATA FORMATS
# ============================================

print("\nSample kmDriven values:")
print(df["kmDriven"].dropna().head(10).to_string(index=False))

print("\nSample AskPrice values:")
print(df["AskPrice"].head(10).to_string(index=False))

print("\nSample PostedDate values:")
print(df["PostedDate"].head(10).to_string(index=False))

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

# clean kmDriven column by removing commas and " km" suffix, then convert to numeric
df["kmDriven"] = (
    df["kmDriven"]
    .str.replace(",", "", regex=False)
    .str.replace(" km", "", regex=False)
)

df["kmDriven"] = pd.to_numeric(df["kmDriven"], errors="coerce")


# clean AskPrice column by removing "₹" symbol and commas, then convert to numeric
df["AskPrice"] = (
    df["AskPrice"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
) 

df["AskPrice"] = pd.to_numeric(df["AskPrice"], errors="coerce")


print("\nData types after conversion:")
print(df[["kmDriven", "AskPrice"]].dtypes)

# missing values after conversion
print("\nMissing kmDriven after conversion:")
print(df["kmDriven"].isna().sum())

print("\nMissing AskPrice after conversion:")
print(df["AskPrice"].isna().sum())


# Remove rows where kmDriven is missing
df = df.dropna(subset=["kmDriven"])

print("\nShape after removing missing kmDriven:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())


# Convert PostedDate to month/year period
df["PostedDate"] = pd.to_datetime(
    df["PostedDate"],
    format="%b-%y"
).dt.to_period("M")

print("\nPostedDate after conversion:")
print(df["PostedDate"].head())

print("\nPostedDate datatype:")
print(df["PostedDate"].dtype)

# ============================================
# 6. CHECK NUMERICAL RANGES
# ============================================
print("\nNumerical Statistics:")

print("\nYear:")
print(df["Year"].describe())

print("\nAge:")
print(df["Age"].describe())

print("\nkmDriven:")
print(df["kmDriven"].describe())

print("\nAskPrice:")
print(df["AskPrice"].describe())


print("\nImpossible / suspicious values:")

print("\nNegative Age:")
print((df["Age"] < 0).sum())

print("\nNegative kmDriven:")
print((df["kmDriven"] < 0).sum())

print("\nZero kmDriven:")
print((df["kmDriven"] == 0).sum())

print("\nZero or negative AskPrice:")
print((df["AskPrice"] <= 0).sum())

print("\nFuture model years:")
print((df["Year"] > 2024).sum())

print("\nYear and Age relationship:")
print(df[["Year", "Age"]].head(20).to_string(index=False))

print("\nYear + Age:")
print((df["Year"] + df["Age"]).value_counts().sort_index())


# Exploratory Data Analysis (EDA)

print("\nNumeric columns summary:")
print(df[["Year", "Age", "kmDriven", "AskPrice"]].describe())


import matplotlib.pyplot as plt

df[["Year", "Age", "kmDriven", "AskPrice"]].hist(
    figsize=(12, 8),
    bins=30
)

plt.tight_layout()
plt.show()

# Scatter plot of Age vs AskPrice

plt.figure(figsize=(8, 5))

plt.scatter(df["Age"], df["AskPrice"])

plt.xlabel("Age")
plt.ylabel("AskPrice")
plt.title("Age vs AskPrice")

plt.show()


 # Bar plot of top 10 car brands by number of listings
brand_counts = df["Brand"].value_counts().head(10)

print(brand_counts)

plt.figure(figsize=(10, 6))

brand_counts.sort_values().plot(kind="barh")

plt.xlabel("Number of Cars")
plt.ylabel("Brand")
plt.title("Top 10 Car Brands by Number of Listings")

plt.tight_layout()
plt.show()

brand_price = df.groupby("Brand")["AskPrice"].mean().sort_values(ascending=False)

print(brand_price)

# Bar plot of top 10 car brands by average ask price

top_brand_price = brand_price.head(10)

plt.figure(figsize=(10, 6))

top_brand_price.sort_values().plot(kind="barh")

plt.xlabel("Average AskPrice")
plt.ylabel("Brand")
plt.title("Top 10 Car Brands by Average Asking Price")

plt.tight_layout()
plt.show()

# Box plot of AskPrice by Transmission

plt.figure(figsize=(8, 6))

df.boxplot(column="AskPrice", by="Transmission")

plt.xlabel("Transmission")
plt.ylabel("AskPrice")
plt.title("AskPrice Distribution by Transmission")
plt.suptitle("")  # removes pandas' extra title

plt.tight_layout()
plt.show()

corr = df[["Year", "Age", "kmDriven", "AskPrice"]].corr()

print(corr)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.show()



# reports

print(df[["Year", "Age", "kmDriven", "AskPrice"]].describe())

print("Brand:")
print(df["Brand"].value_counts().head(10))

print("\nTransmission:")
print(df["Transmission"].value_counts())

print("\nFuel Type:")
print(df["FuelType"].value_counts())

print("\nOwner:")
print(df["Owner"].value_counts())

print(
    df.groupby("Brand")["AskPrice"]
      .agg(["count", "mean", "median"])
      .sort_values("mean", ascending=False)
      .head(10)
)

print(
    df.groupby("Transmission")["AskPrice"]
      .agg(["count", "mean", "median"])
)

print(
    df[["Year", "Age", "kmDriven", "AskPrice"]]
      .corr()
      .round(2)
)
