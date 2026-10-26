"""Classification models and training/evaluation helpers."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from .evaluation import eval_classifier

CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000),
    "SGD Classifier": lambda: SGDClassifier(
        class_weight="balanced", loss="squared_hinge", tol=1e-5, max_iter=10000),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, n_jobs=-1, max_depth=30, max_features=50),
    "MLP (Neural Network)": lambda: MLPClassifier(
        activation="logistic", hidden_layer_sizes=(10, 10, 5, 3)),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=200),
    "Gaussian Naive Bayes": lambda: GaussianNB(),
    "Linear SVC": lambda: LinearSVC(
        class_weight="balanced", dual=False, penalty="l2", loss="squared_hinge"),
}

RUN_CLASSIFIERS = {
    "Logistic Regression": True,
    "SGD Classifier": False,
    "Random Forest": True,
    "MLP (Neural Network)": True,
    "AdaBoost": False,
    "Gaussian Naive Bayes": False,
    "Linear SVC": True,
}

def run_classifier(name, train_x, train_y, test_x, test_y):
    """Fit one configured classifier and return metrics and predictions."""
    model = CLASSIFIERS[name]()
    started = time.time()
    model.fit(train_x, train_y)
    fit_time = time.time() - started
    train_metrics, train_pred, _ = eval_classifier(model, train_x, train_y)
    test_metrics, test_pred, test_score = eval_classifier(model, test_x, test_y)
    return {
        "model": model,
        "params": repr(model),
        "fit_time_s": fit_time,
        "train": train_metrics,
        "test": test_metrics,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
        "test_score": test_score,
    }
