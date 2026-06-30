# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report
# )


# def evaluate_model(model, X_test, y_test):

#     predictions = model.predict(X_test)
#     probabilities = model.predict_proba(X_test)[:, 1]

#     print("=" * 60)
#     print("Model Evaluation")
#     print("=" * 60)

#     print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
#     print(f"Precision: {precision_score(y_test, predictions):.4f}")
#     print(f"Recall   : {recall_score(y_test, predictions):.4f}")
#     print(f"F1 Score : {f1_score(y_test, predictions):.4f}")
#     print(f"ROC AUC  : {roc_auc_score(y_test, probabilities):.4f}")

#     print("\nConfusion Matrix")
#     print(confusion_matrix(y_test, predictions))

#     print("\nClassification Report")
#     print(classification_report(y_test, predictions))





from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, probability)

    print("=" * 60)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    return accuracy