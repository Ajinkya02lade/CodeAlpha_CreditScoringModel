# import pandas as pd
# from sklearn.preprocessing import LabelEncoder


# def load_data(path):
#     """
#     Load dataset from CSV file.
#     """
#     return pd.read_csv(path)


# def preprocess_data(df):
#     """
#     Handle missing values and encode categorical columns.
#     """

#     # Fill missing values
#     df["person_emp_length"] = df["person_emp_length"].fillna(
#         df["person_emp_length"].median()
#     )

#     df["loan_int_rate"] = df["loan_int_rate"].fillna(
#         df["loan_int_rate"].median()
#     )

#     # Encode categorical columns
#     categorical_columns = [
#         "person_home_ownership",
#         "loan_intent",
#         "loan_grade",
#         "cb_person_default_on_file"
#     ]

#     encoders = {}

#     for col in categorical_columns:
#         encoder = LabelEncoder()
#         df[col] = encoder.fit_transform(df[col])
#         encoders[col] = encoder

#     return df, encoders



import pandas as pd

from sklearn.preprocessing import LabelEncoder


def load_dataset(path):
    return pd.read_csv(path)


def preprocess(df):

    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df["loan_int_rate"].median()
    )

    categorical_columns = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return X, y