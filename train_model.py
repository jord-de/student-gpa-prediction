import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# Load the dataset

df = pd.read_csv(
    "dataset/student_social_media.csv"
)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)

print(df.head())

# Remove Student ID because it is an identifier

df = df.drop(columns=["Student ID"])

# Select input features

X = df.drop(columns=["GPA"])

# Select target variable

y = df["GPA"]

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget variable:")
print(y.name)

# Identify numerical columns

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify categorical columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# Process numerical columns

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Process categorical columns

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDATASET SPLIT")
print("---------------------")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create Linear Regression pipeline

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train Linear Regression

linear_model.fit(X_train, y_train)

print("\nLinear Regression training completed!")

# Create Random Forest pipeline

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train Random Forest

random_forest_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Function to evaluate models

def evaluate_model(model, model_name):

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test, predictions
    )

    mse = mean_squared_error(
        y_test, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test, predictions
    )

    print(f"\n{model_name} RESULTS")
    print("-------------------------")

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }


# Evaluate Linear Regression

linear_results = evaluate_model(
    linear_model,
    "Linear Regression"
)


# Evaluate Random Forest

forest_results = evaluate_model(
    random_forest_model,
    "Random Forest"
)

# Compare model results

results = pd.DataFrame([
    linear_results,
    forest_results
])

print("\nMODEL PERFORMANCE COMPARISON")
print("============================")

print(results.to_string(index=False))

# Save results

results.to_csv(
    "model_comparison.csv",
    index=False
)

# Select the best-performing model

if forest_results["R2 Score"] > linear_results["R2 Score"]:
    best_model = random_forest_model
    best_model_name = "Random Forest"

elif forest_results["R2 Score"] < linear_results["R2 Score"]:
    best_model = linear_model
    best_model_name = "Linear Regression"

else:
    if forest_results["MAE"] <= linear_results["MAE"]:
        best_model = random_forest_model
        best_model_name = "Random Forest"
    else:
        best_model = linear_model
        best_model_name = "Linear Regression"


# Create model folder

os.makedirs("model", exist_ok=True)

# Save the selected model

joblib.dump(
    best_model,
    "model/best_student_performance_model.pkl"
)

print("\nBEST MODEL SELECTED")
print("-------------------")

print("Selected model:", best_model_name)

print("Model saved successfully!")