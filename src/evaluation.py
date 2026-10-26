"""Metrics, checkpointing, and loan-selection strategy."""
import json
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, roc_auc_score,
    average_precision_score, r2_score, mean_squared_error
)

def metrics_from_pred(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["Default", "Paid Off"],
        output_dict=True, zero_division=0
    )
    report.pop("accuracy", None)
    return {
        "confusion_matrix": cm.tolist(),
        "report": report,
        "accuracy": float(accuracy_score(y, pred)),
    }

def eval_classifier(model, X, y):
    pred = model.predict(X)
    result = metrics_from_pred(y, pred)
    score = None
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
    result["roc_auc"] = (
        float(roc_auc_score(y, score)) if score is not None else None
    )
    result["avg_precision"] = (
        float(average_precision_score(y, score)) if score is not None else None
    )
    return result, pred, score

def eval_regressor(model, X, y):
    pred = model.predict(X)
    mse = float(mean_squared_error(y, pred))
    return {
        "r2": float(r2_score(y, pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }, pred

def save_checkpoint(path, meta, classification, regression, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta,
            "classification": classification,
            "regression": regression,
            "extra": extra or {},
        }, f, indent=2, default=str)

def thin(values, n=2000):
    values = np.asarray(values)
    return values if len(values) <= n else values[
        np.linspace(0, len(values) - 1, n).astype(int)
    ]

def investment_strategy(train_pred, train_y, test_pred, test_y,
                        target=0.156, min_k=100, stake=25):
    """Notebook strategy: find a training cutoff, then evaluate selected test loans."""
    ordered = train_y[np.argsort(-train_pred)] - 1.0
    n = len(ordered)
    if n < min_k:
        return None
    ks = np.arange(min_k, n + 1)
    net_profit = np.cumsum(ordered)[min_k - 1:] / ks
    below = np.where(net_profit < target)[0]
    if len(below) == 0:
        return None
    i = int(below[0])
    cutoff = float(-np.sort(-train_pred)[i])
    selected_train = train_pred >= cutoff
    selected_test = test_pred >= cutoff
    invested = stake * int(selected_test.sum())
    profit = float((stake * (test_y[selected_test] - 1.0)).sum())
    return {
        "cutoff_index": i,
        "cutoff_prediction": cutoff,
        "target_return": target,
        "train_pct_loans_selected": 100.0 * float(selected_train.mean()),
        "train_avg_return_selected": (
            float((train_y[selected_train] - 1.0).mean())
            if selected_train.any() else None
        ),
        "test_n_loans_selected": int(selected_test.sum()),
        "test_pct_loans_selected": 100.0 * float(selected_test.mean()),
        "test_invested_usd": float(invested),
        "test_profit_usd": profit,
        "test_return_on_invested": profit / invested if invested else None,
    }
