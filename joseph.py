# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully.\n")
    print("First five rows of the dataset:")
    print(df.head())

    # Check structure
    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    # No missing values in Iris dataset; if any, handle here:
    # df = df.dropna() or df.fillna(...)

except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
print("\nStatistical Summary:")
print(df.describe())

# Group by species and get mean of numerical columns
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Optional: Observations
print("\nObservations:")
print("- Setosa generally has smaller petal dimensions.")
print("- Virginica species tends to have the largest petal length and width.")

# Task 3: Data Visualization

# 1. Line chart - simulate trend by cumulative sum (since Iris dataset has no time series)
df['index'] = df.index
cumsum_df = df.groupby('species')[['petal length (cm)']].cumsum()
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    plt.plot(df[df['species'] == species]['index'], 
             cumsum_df[df['species'] == species], 
             label=species)
plt.title('Simulated Petal Length Trend Over Index')
plt.xlabel('Index')
plt.ylabel('Cumulative Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart: Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram of sepal length
plt.figure(figsize=(8, 5))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
