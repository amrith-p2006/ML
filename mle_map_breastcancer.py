import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Column names
columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",

    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

# Load dataset
df = pd.read_csv("C:\\Users\\USER\\OneDrive\\Documents\\C Programming\\ml\\potti\\wdbc.data", header=None, names=columns)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Remove ID column
df = df.drop("id", axis=1)

# Convert diagnosis into numerical values
# M = 1 (Malignant)
# B = 0 (Benign)

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Separate X and y
X = df.drop("diagnosis", axis=1).values

y = df["diagnosis"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


mle_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty=None,
        max_iter=10000
    ))
])

mle_model.fit(X_train, y_train)

y_pred_mle = mle_model.predict(X_test)

print("MLE Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_mle))

# L1 MAP model
l1_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=10000
    ))
])

# Alpha is inversely related to C.
# We test different regularization strengths.
l1_params = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

l1_grid = GridSearchCV(
    l1_pipeline,
    l1_params,
    cv=5,
    scoring="accuracy"
)

l1_grid.fit(X_train, y_train)

best_l1 = l1_grid.best_estimator_

y_pred_l1 = best_l1.predict(X_test)

print("\nMAP with L1 Regularization")
print("Best C:", l1_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_l1))

# L2 MAP model
l2_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=10000
    ))
])

l2_params = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

l2_grid = GridSearchCV(
    l2_pipeline,
    l2_params,
    cv=5,
    scoring="accuracy"
)

l2_grid.fit(X_train, y_train)

best_l2 = l2_grid.best_estimator_

y_pred_l2 = best_l2.predict(X_test)


print("\nMAP with L2 Regularization")
print("Best C:", l2_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_l2))

results = []

models = {
    "MLE": (mle_model, y_pred_mle),
    "MAP - L1": (best_l1, y_pred_l1),
    "MAP - L2": (best_l2, y_pred_l2)
}

for name, (model, y_pred) in models.items():

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# Extract coefficients from each model
mle_coef = mle_model.named_steps["model"].coef_[0]

l1_coef = best_l1.named_steps["model"].coef_[0]

l2_coef = best_l2.named_steps["model"].coef_[0]

# Compare coefficients
coef_df = pd.DataFrame({
    "Feature": columns[2:],
    "MLE": mle_coef,
    "MAP_L1": l1_coef,
    "MAP_L2": l2_coef
})

print("\nParameter Comparison:")
print(coef_df.to_string(index=False))

print("MLE non-zero coefficients:",
      np.sum(mle_coef != 0))

print("L1 non-zero coefficients:",
      np.sum(l1_coef != 0))

print("L2 non-zero coefficients:",
      np.sum(l2_coef != 0))

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xlabel("Estimation Method")
plt.ylabel("Accuracy")
plt.title("MLE vs MAP Logistic Regression")

plt.ylim(0, 1.05)
plt.grid(axis="y")

plt.show()

coef_df.set_index("Feature").plot(
    kind="bar",
    figsize=(14, 6)
)

plt.title("Comparison of MLE and MAP Coefficients")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()