import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

file_path = r"C:\Users\vande\becode\immo-eliza-ml\cleaned_properties.csv"

# Load CSV into a DataFrame
cleaned_properties = pd.read_csv(file_path)

# Count NaNs per column
nan_counts = cleaned_properties.isna().sum()

# Keep only columns with at least one NaN
nan_counts = nan_counts[nan_counts > 0]

# Sort in descending order
nan_counts_sorted = nan_counts.sort_values(ascending=False)

print("Columns with missing values (sorted):\n", nan_counts_sorted)

#Basic distributions for numeric columns
print("\nBasic statistics:\n", cleaned_properties.describe())

#Target (price) distribution
plt.figure(figsize=(8, 5))
sns.histplot(cleaned_properties['price'], kde=True, bins=30)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

#Skewness of price
price_skew = cleaned_properties['price'].skew()
print(f"\nSkewness of price: {price_skew:.2f}")

X = cleaned_properties.drop(columns=["price"])
y = cleaned_properties["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_train.select_dtypes(include=["object", "bool"]).columns

numeric_cols, categorical_cols

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize model
lr_model = LinearRegression()

# Fit on preprocessed training data
lr_model.fit(X_train_processed, y_train)

y_pred = lr_model.predict(X_test_processed)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
