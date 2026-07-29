import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(y_true, y_scores, y_pred):

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "mcc": np.nan,
            "auc": np.nan,
            "confusion_matrix": np.array([[0, 0], [0, 0]])
        }

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mcc": mcc,
        "auc": auc,
        "confusion_matrix": cm,
    }