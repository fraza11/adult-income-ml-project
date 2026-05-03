
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train_df = pd.read_csv(DATA_URL, names=columns, na_values=" ?", skipinitialspace=True)
test_df = pd.read_csv(TEST_URL, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

df = pd.concat([train_df, test_df], ignore_index=True)
df["target"] = (df["income"] == ">50K").astype(int)
df = df.drop(columns=["income"])

print(df.shape)
print(df.head())
print(df.isna().sum())
print(df["target"].value_counts(normalize=True))

X = df.drop(columns=["target"])
y = df["target"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=18, random_state=42, class_weight="balanced"))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    })
    print("\n" + name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
print(results_df)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [12, 18, None],
    "model__min_samples_leaf": [1, 3]
}

rf_pipeline = models["Random Forest"]
grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best RF parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

df["target"].value_counts().plot(kind="bar", title="Class Distribution")
plt.xlabel("Income target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

results_df.set_index("Model")[["Accuracy", "F1", "ROC_AUC"]].plot(kind="bar", ylim=(0, 1), title="Model Performance")
plt.tight_layout()
plt.savefig("model_performance.png", dpi=150)
plt.close()
