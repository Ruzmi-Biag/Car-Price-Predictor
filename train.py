
import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
file_path = "data/car_data.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset not found!")
    print("Please put car_data.csv inside the data folder.")
    exit()

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset Shape:", data.shape)
print("\nDataset Columns:")
print(data.columns.tolist())

# Step 2: Identify target column
target_options = [
    "selling_price",
    "selling price",
    "sellingprice",
    "price",
    "car_price",
    "car price"
]

target_column = None

for column in data.columns:
    if column.strip().lower() in target_options:
        target_column = column
        break

if target_column is None:
    print("\nERROR: Price column not found!")
    print("Please check the dataset column names.")
    exit()

print("\nTarget Column:", target_column)

# Step 3: Clean target values
def clean_price(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace(",", "").replace("₹", "").strip()
    return pd.to_numeric(value, errors="coerce")

data[target_column] = data[target_column].apply(clean_price)

data = data.dropna(subset=[target_column])
data = data.dropna(axis=1, how="all")
data = data.dropna(axis=0, how="all")

# Step 4: Separate input and output
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert numeric-looking columns
for column in X.columns:
    if X[column].dtype == "object":
        converted = pd.to_numeric(X[column], errors="coerce")
        if converted.notna().mean() >= 0.95:
            X[column] = converted

# Remove columns with no useful values
X = X.loc[:, X.notna().any()]

if X.shape[1] == 0:
    print("ERROR: No useful input columns found.")
    exit()

# Step 5: Identify column types
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# Step 6: Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_columns),
    ("categorical", categorical_transformer, categorical_columns)
])

# Step 7: Create Machine Learning model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# Step 8: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 9: Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Step 10: Make predictions
predictions = model.predict(X_test)

# Step 11: Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-----------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Step 12: Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/car_price_model.pkl")

# Save input column information
schema = {
    "columns": X.columns.tolist(),
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns
}

with open("models/input_schema.json", "w") as file:
    json.dump(schema, file, indent=4)

# Save performance metrics
metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nSUCCESS! Model trained and saved.")
print("Model location: models/car_price_model.pkl")
