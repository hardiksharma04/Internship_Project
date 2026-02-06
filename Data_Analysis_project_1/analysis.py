import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Basic statistics
print("\nAverage Scores:")
print(df[['math score', 'reading score', 'writing score']].mean())

# Bar Chart - Average Scores
df[['math score', 'reading score', 'writing score']].mean().plot(kind='bar')
plt.title("Average Student Scores")
plt.ylabel("Score")
plt.xlabel("Subjects")
plt.show()

# Scatter Plot - Math vs Reading
plt.scatter(df['math score'], df['reading score'])
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Math vs Reading Score")
plt.show()

# Heatmap - Correlation
corr_matrix = df[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(6,4))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Correlation')
plt.xticks([0, 1, 2], ['math score', 'reading score', 'writing score'])
plt.yticks([0, 1, 2], ['math score', 'reading score', 'writing score'])
plt.title("Correlation Heatmap")
plt.show()

# ===== INSIGHTS AND OBSERVATIONS =====
print("\n" + "="*50)
print("INSIGHTS AND OBSERVATIONS")
print("="*50)

# Average scores analysis
avg_scores = df[['math score', 'reading score', 'writing score']].mean()
print("\n1. AVERAGE SCORES:")
print(f"   - Math Score: {avg_scores['math score']:.2f}")
print(f"   - Reading Score: {avg_scores['reading score']:.2f}")
print(f"   - Writing Score: {avg_scores['writing score']:.2f}")

# Highest and lowest performing subjects
print(f"\n2. SUBJECT PERFORMANCE:")
print(f"   - Highest Average: {avg_scores.idxmax()} ({avg_scores.max():.2f})")
print(f"   - Lowest Average: {avg_scores.idxmin()} ({avg_scores.min():.2f})")

# Correlation analysis
print("\n3. CORRELATION ANALYSIS:")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        print(f"   - {col1} vs {col2}: {corr_value:.3f}")

# Score distribution insights
print("\n4. SCORE DISTRIBUTION:")
print(f"   - Math Score Range: {df['math score'].min():.0f} - {df['math score'].max():.0f}")
print(f"   - Reading Score Range: {df['reading score'].min():.0f} - {df['reading score'].max():.0f}")
print(f"   - Writing Score Range: {df['writing score'].min():.0f} - {df['writing score'].max():.0f}")

print("\n" + "="*50)
