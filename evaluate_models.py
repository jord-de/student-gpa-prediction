
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
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

# =====================================
# 1. LOAD DATASET
# =====================================

df = pd.read_csv("dataset/student_social_media.csv")

df = df.drop(columns=["Student ID"])

X = df.drop(columns=["GPA"])
y = df["GPA"]

print("Dataset loaded successfully!")
print("Total records:", len(df))


# =====================================
# 2. SPLIT DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================
# 3. PREPROCESSING
# =====================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


def create_preprocessor():

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


# =====================================
# 4. CREATE MODELS
# =====================================

models = {

    "Linear Regression": Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", LinearRegression())
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
}


# =====================================
# 5. CROSS-VALIDATION
# =====================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_results = {}

print("\nCROSS-VALIDATION RESULTS")
print("========================")


for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    cv_results[name] = {
        "CV MAE": -scores["test_mae"].mean(),
        "CV RMSE": -scores["test_rmse"].mean(),
        "CV R2": scores["test_r2"].mean(),
        "Training CV RMSE": -scores["train_rmse"].mean()
    }

    print(f"\n{name}")
    print("CV MAE:", round(cv_results[name]["CV MAE"], 4))
    print("CV RMSE:", round(cv_results[name]["CV RMSE"], 4))
    print("CV R2:", round(cv_results[name]["CV R2"], 4))


# =====================================
# 6. TRAIN AND TEST BOTH MODELS
# =====================================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    mae = mean_absolute_error(
        y_test, test_predictions
    )

    rmse = test_rmse

    r2 = r2_score(
        y_test, test_predictions
    )

    results.append({
        "Model": name,
        "CV MAE": cv_results[name]["CV MAE"],
        "CV RMSE": cv_results[name]["CV RMSE"],
        "CV R2": cv_results[name]["CV R2"],
        "Training RMSE": train_rmse,
        "Testing MAE": mae,
        "Testing RMSE": rmse,
        "Testing R2": r2
    })

    print(f"\n{name} TEST RESULTS")
    print("-----------------------")
    print("Training RMSE:", round(train_rmse, 4))
    print("Testing MAE:", round(mae, 4))
    print("Testing RMSE:", round(rmse, 4))
    print("Testing R2:", round(r2, 4))


# =====================================
# 7. COMPARE BOTH MODELS
# =====================================

comparison = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print("======================")

print(comparison.round(4).to_string(index=False))

os.makedirs("visualizations", exist_ok=True)
os.makedirs("model", exist_ok=True)

comparison.to_csv(
    "model_comparison_phase4.csv",
    index=False
)


# =====================================
# 8. SELECT BEST MODEL
# =====================================

# Select using cross-validation RMSE,
# not the final test results.

best_name = comparison.loc[
    comparison["CV RMSE"].idxmin(),
    "Model"
]

best_model = models[best_name]

# Fit the selected model on the full
# training set before saving.

best_model.fit(X_train, y_train)

joblib.dump(
    best_model,
    "model/best_student_performance_model.pkl"
)

print("\nBEST MODEL")
print("==========")
print("Selected:", best_name)
print("Saved successfully!")


# =====================================
# 9. PREDICTION VISUALIZATION
# =====================================

predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=predictions
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")

plt.title("Actual GPA vs Predicted GPA")

plt.savefig(
    "visualizations/actual_vs_predicted.png"
)

plt.show()


# =====================================
# 10. RESIDUAL ANALYSIS
# =====================================

residuals = y_test - predictions

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=predictions,
    y=residuals
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted GPA")
plt.ylabel("Prediction Error")

plt.title("Residual Analysis")

plt.savefig(
    "visualizations/residual_analysis.png"
)

plt.show()


# =====================================
# 11. FEATURE IMPORTANCE
# =====================================

regressor = best_model.named_steps["regressor"]

feature_names = (
    best_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

if best_name == "Random Forest":

    importance_values = regressor.feature_importances_

    importance_label = "Random Forest Feature Importance"

else:

    importance_values = np.abs(regressor.coef_)

    importance_label = "Absolute Linear Regression Coefficients"


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print("=========================")

print(importance_df.head(10).to_string(index=False))

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title(importance_label)

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png"
)

plt.show()


print("\nPHASE 4 COMPLETED SUCCESSFULLY!")
