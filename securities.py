import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("securities.csv")

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(df.head())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Sector Analysis
print("\nSector Analysis:")
sector_counts = df['GICS Sector'].value_counts()
print(sector_counts)

# Plot Sector Analysis
plt.figure(figsize=(10, 6))
sns.countplot(y='GICS Sector', data=df, order=sector_counts.index)
plt.title('Distribution of Companies by GICS Sector')
plt.xlabel('Number of Companies')
plt.ylabel('GICS Sector')
plt.show()

# Geographical Analysis
print("\nGeographical Analysis:")
city_counts = df['Address of Headquarters'].value_counts().head(10)
print(city_counts)

# Plot Geographical Analysis
plt.figure(figsize=(10, 6))
city_counts.plot(kind='bar', color='skyblue')
plt.title('Top 10 Headquarters Locations with Most Companies')
plt.xlabel('City')
plt.ylabel('Number of Companies')
plt.xticks(rotation=45)
plt.show()

# Trend Analysis
print("\nTrend Analysis:")
df['Date first added'] = pd.to_datetime(df['Date first added'])
yearly_counts = df['Date first added'].dt.year.value_counts().sort_index()
print(yearly_counts)

# Plot Trend Analysis
plt.figure(figsize=(10, 6))
yearly_counts.plot(kind='line', marker='o')
plt.title('Number of Companies Added Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Companies')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Correlation Analysis
print("\nCorrelation Analysis:")
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# Plot Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Feature Importance - Not implemented without target variable for prediction

# Additional Visualization - Histogram of CIK
plt.figure(figsize=(10, 6))
sns.histplot(df['CIK'], bins=20, kde=True)
plt.title('Histogram of CIK')
plt.xlabel('CIK')
plt.ylabel('Frequency')
plt.show()
