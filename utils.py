# import joblib


# def save_model(model, path):
#     """
#     Save trained model.
#     """
#     joblib.dump(model, path)


# def load_model(path):
#     """
#     Load trained model.
#     """
#     return joblib.load(path)


# def save_scaler(scaler, path):
#     """
#     Save StandardScaler.
#     """
#     joblib.dump(scaler, path)


# def load_scaler(path):
#     """
#     Load StandardScaler.
#     """
#     return joblib.load(path)





import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_scaler(scaler, path):
    joblib.dump(scaler, path)


def load_scaler(path):
    return joblib.load(path)