# import pandas as pd
# import joblib

# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     classification_report,
#     confusion_matrix
# )

# # ===============================
# # Load Dataset
# # ===============================

# df = pd.read_csv("dataset/credit_risk_dataset.csv")

# # ===============================
# # Handle Missing Values
# # ===============================

# # df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
# # df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

# df["person_emp_length"] = df["person_emp_length"].fillna(
#     df["person_emp_length"].median()
# )

# df["loan_int_rate"] = df["loan_int_rate"].fillna(
#     df["loan_int_rate"].median()
# )

# # ===============================
# # Encode Categorical Columns
# # ===============================

# encoder = LabelEncoder()

# categorical_columns = [
#     "person_home_ownership",
#     "loan_intent",
#     "loan_grade",
#     "cb_person_default_on_file"
# ]

# for col in categorical_columns:
#     df[col] = encoder.fit_transform(df[col])

# # ===============================
# # Features and Target
# # ===============================

# X = df.drop("loan_status", axis=1)
# y = df["loan_status"]

# # ===============================
# # Train Test Split
# # ===============================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# # ===============================
# # Feature Scaling
# # ===============================

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Save scaler
# joblib.dump(scaler, "models/scaler.pkl")

# # ===============================
# # Models
# # ===============================

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": RandomForestClassifier(
#         n_estimators=100,
#         random_state=42
#     )
# }

# best_model = None
# best_accuracy = 0

# # ===============================
# # Training
# # ===============================

# for name, model in models.items():

#     print("\n" + "=" * 60)
#     print(name)
#     print("=" * 60)

#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)
#     probabilities = model.predict_proba(X_test)[:, 1]

#     accuracy = accuracy_score(y_test, predictions)
#     precision = precision_score(y_test, predictions)
#     recall = recall_score(y_test, predictions)
#     f1 = f1_score(y_test, predictions)
#     roc_auc = roc_auc_score(y_test, probabilities)

#     print(f"Accuracy : {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall   : {recall:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#     print(f"ROC AUC  : {roc_auc:.4f}")

#     print("\nConfusion Matrix")
#     print(confusion_matrix(y_test, predictions))

#     print("\nClassification Report")
#     print(classification_report(y_test, predictions))

#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = model

# # ===============================
# # Save Best Model
# # ===============================

# joblib.dump(best_model, "models/credit_model.pkl")

# print("\n" + "=" * 60)
# print("Best Model Saved Successfully!")
# print("Accuracy:", round(best_accuracy, 4))
# print("=" * 60)















import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset/credit_risk_dataset.csv")

# Features & target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "models/credit_model.pkl")

print("Model saved successfully!")