import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

# ==================== 1. LOAD AND EXPLORE DATA ====================
print("=" * 60)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("=" * 60)

# Load dataset
df = pd.read_csv("Housing.csv")

# Display dataset info
print("\n1. DATASET OVERVIEW:")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nStatistical Summary:")
print(df.describe())
print(f"\nMissing Values:")
print(df.isnull().sum())

# ==================== 2. DATA PREPROCESSING ====================
print("\n" + "=" * 60)
print("2. DATA PREPROCESSING")
print("=" * 60)

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']

print(f"\nEncoding categorical variables: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    if col != 'furnishingstatus':  # Binary columns
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    else:  # Multi-class categorical
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("Categorical encoding complete")

# ==================== 3. FEATURE SELECTION & SPLITTING ====================
print("\n" + "=" * 60)
print("3. FEATURE SELECTION & DATA SPLITTING")
print("=" * 60)

# Select all features except price (target)
feature_cols = [col for col in df_processed.columns if col != 'price']
X = df_processed[feature_cols]
y = df_processed['price']

print(f"\nFeatures used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print(f"\nTarget variable: price")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split dataset into training & testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ==================== 4. MODEL TRAINING ====================
print("\n" + "=" * 60)
print("4. MODEL TRAINING")
print("=" * 60)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("✓ Linear Regression model trained successfully")

# Display model coefficients
print(f"\nModel Coefficients:")
print(f"{'Feature':<25} {'Coefficient':>15}")
print("-" * 42)
for feature, coef in zip(feature_cols, model.coef_):
    print(f"{feature:<25} {coef:>15,.2f}")
print(f"{'Intercept':<25} {model.intercept_:>15,.2f}")

# ==================== 5. MODEL PREDICTION & EVALUATION ====================
print("\n" + "=" * 60)
print("5. MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\nTraining Set Metrics:")
print(f"  Mean Squared Error (MSE):   {mse_train:,.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_train:,.2f}")
print(f"  Mean Absolute Error (MAE):  {mae_train:,.2f}")
print(f"  R² Score:                   {r2_train:.4f}")

print("\nTesting Set Metrics:")
print(f"  Mean Squared Error (MSE):   {mse_test:,.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:,.2f}")
print(f"  Mean Absolute Error (MAE):  {mae_test:,.2f}")
print(f"  R² Score:                   {r2_test:.4f}")

# ==================== 6. SAMPLE PREDICTIONS ====================
print("\n" + "=" * 60)
print("6. SAMPLE PREDICTIONS")
print("=" * 60)
print(f"\n{'Actual Price':<20} {'Predicted Price':<20} {'Error ($)':<20}")
print("-" * 60)
for i in range(min(5, len(y_test))):
    error = y_test.iloc[i] - y_pred_test[i]
    print(f"{y_test.iloc[i]:<20,.0f} {y_pred_test[i]:<20,.0f} {error:<20,.0f}")

# ==================== 7. VISUALIZATIONS ====================
print("\n" + "=" * 60)
print("7. GENERATING VISUALIZATIONS")
print("=" * 60)

plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted Prices (Test Set)
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price", fontsize=11)
plt.ylabel("Predicted Price", fontsize=11)
plt.title("Actual vs Predicted House Prices (Test Set)", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals (Prediction Errors)
plt.subplot(2, 2, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Price", fontsize=11)
plt.ylabel("Residuals", fontsize=11)
plt.title("Residual Plot (Test Set)", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 3: Distribution of Residuals
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Residuals", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.title("Distribution of Residuals", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Model Coefficients
plt.subplot(2, 2, 4)
coefficients = pd.Series(model.coef_, index=feature_cols).sort_values()
coefficients.plot(kind='barh', color=['red' if x < 0 else 'green' for x in coefficients])
plt.xlabel("Coefficient Value", fontsize=11)
plt.title("Feature Importance (Coefficients)", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('house_price_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'house_price_prediction_analysis.png'")
plt.show()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
