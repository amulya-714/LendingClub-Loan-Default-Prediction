"""Regression models and training/evaluation helpers."""
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from .evaluation import eval_regressor

REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(n_jobs=-1),
    "Ridge Regression": lambda: Ridge(),
    "Linear SVR": lambda: LinearSVR(
        loss="squared_epsilon_insensitive", dual=False),
    "MLP Regressor (20,10,5,3)": lambda: MLPRegressor(
        hidden_layer_sizes=(20, 10, 5, 3), alpha=0.008,
        learning_rate="adaptive", max_iter=1000),
    "Random Forest Regressor": lambda: RandomForestRegressor(
        n_estimators=100, min_samples_split=0.01, n_jobs=-1),
    "Kernel Ridge": lambda: KernelRidge(),
}
for depth in range(4, 11):
    REGRESSORS[f"RF Regressor (max_depth={depth})"] = (
        lambda depth=depth: RandomForestRegressor(
            n_estimators=100, max_depth=depth, n_jobs=-1)
    )

RUN_REGRESSORS = {
    "Linear Regression": True,
    "Ridge Regression": True,
    "Linear SVR": False,
    "MLP Regressor (20,10,5,3)": True,
    "Random Forest Regressor": True,
    **{f"RF Regressor (max_depth={d})": True for d in range(4, 11)},
    "Kernel Ridge": False,
}

STRATEGY_MODEL = "RF Regressor (max_depth=10)"
STRATEGY_TARGET_RETURN = 0.156

def run_regressor(name, train_x, train_y, test_x, test_y):
    """Fit one configured regressor and return metrics and predictions."""
    model = REGRESSORS[name]()
    started = time.time()
    model.fit(train_x, train_y)
    fit_time = time.time() - started
    train_metrics, train_pred = eval_regressor(model, train_x, train_y)
    test_metrics, test_pred = eval_regressor(model, test_x, test_y)
    return {
        "model": model,
        "params": repr(model),
        "fit_time_s": fit_time,
        "train": train_metrics,
        "test": test_metrics,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
    }
