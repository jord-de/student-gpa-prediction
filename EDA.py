import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for charts
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/student_social_media.csv")

# Display dataset
print("\nFIRST FIVE RECORDS")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())
plt.figure(figsize=(9, 5))

sns.histplot(
    df["GPA"],
    bins=20,
    kde=True
)

plt.title("Distribution of Students' GPA")
plt.xlabel("GPA")
plt.ylabel("Number of Students")

plt.savefig("visualizations/gpa_distribution.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="Daily Social Media Hours",
    y="GPA"
)

plt.title("Social Media Usage vs Student GPA")
plt.xlabel("Daily Social Media Hours")
plt.ylabel("GPA")

plt.savefig("visualizations/social_media_vs_gpa.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.regplot(
    data=df,
    x="Study Hours Per Day",
    y="GPA"
)

plt.title("Study Hours vs Student GPA")
plt.xlabel("Study Hours Per Day")
plt.ylabel("GPA")

plt.savefig("visualizations/study_hours_vs_gpa.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="Sleep Hours Per Day",
    y="GPA"
)

plt.title("Sleep Hours vs Student GPA")
plt.xlabel("Sleep Hours Per Day")
plt.ylabel("GPA")

plt.savefig("visualizations/sleep_hours_vs_gpa.png")

plt.show()

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Social Media Platform",
    y="GPA",
    errorbar=None
)

plt.title("Average GPA by Social Media Platform")
plt.xlabel("Social Media Platform")
plt.ylabel("Average GPA")

plt.savefig("visualizations/gpa_by_platform.png")

plt.show()

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])

# Calculate correlations
correlation = numeric_df.corr()

# Display correlation values
print("\nCORRELATION MATRIX")
print(correlation)

# Create heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Student Variables")

plt.savefig("visualizations/correlation_heatmap.png")

plt.show()