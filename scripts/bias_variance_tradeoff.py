import numpy as np
from numpy.linalg import inv


def create_regressors(x, max_order):
    """Create polynomial regressor matrices from order 1 up to max_order."""
    x = np.asarray(x, dtype=float).ravel()
    regressors = []
    for order in range(1, max_order + 1):
        current = np.zeros((len(x), order + 1))
        current[:, 0] = 1.0
        for i in range(1, order + 1):
            current[:, i] = np.power(x, i)
        regressors.append(current)
    return regressors


def estimate_weights(p_list, y, reg=None):
    """Estimate polynomial regression weights for each regressor matrix."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    weights = []
    if reg is None:
        for p in p_list:
            if p.shape[1] > p.shape[0]:
                w = p.T @ inv(p @ p.T) @ y
            else:
                w = inv(p.T @ p) @ p.T @ y
            weights.append(w)
    else:
        for p in p_list:
            w = inv(p.T @ p + reg * np.eye(p.shape[1])) @ p.T @ y
            weights.append(w)
    return weights


def perform_prediction(p_list, w_list):
    """Predict y values for each polynomial order."""
    n_samples = p_list[0].shape[0]
    max_order = len(p_list)
    y_predict_mat = np.zeros((n_samples, max_order))
    for order in range(max_order):
        y_predict_mat[:, order] = (p_list[order] @ w_list[order]).ravel()
    return y_predict_mat


def run_bias_variance_tradeoff(x, y, xt, yt, max_order=6, reg=None):
    """Compute training and test MSE curves for polynomial orders 1..max_order."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xt = np.asarray(xt, dtype=float)
    yt = np.asarray(yt, dtype=float)

    p_train_list = create_regressors(x, max_order)
    p_test_list = create_regressors(xt, max_order)
    w_list = estimate_weights(p_train_list, y, reg)
    y_train_pred = perform_prediction(p_train_list, w_list)
    y_test_pred = perform_prediction(p_test_list, w_list)

    train_mse = np.zeros(max_order)
    test_mse = np.zeros(max_order)
    train_loss = None if reg is None else np.zeros(max_order)

    for i in range(max_order):
        train_mse[i] = np.mean((y_train_pred[:, i] - y) ** 2)
        test_mse[i] = np.mean((y_test_pred[:, i] - yt) ** 2)
        if reg is not None:
            train_sse = np.sum((y_train_pred[:, i] - y) ** 2)
            train_loss[i] = train_sse + reg * np.sum(w_list[i] ** 2)

    return {
        "weights": w_list,
        "train_predictions": y_train_pred,
        "test_predictions": y_test_pred,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_loss": train_loss,
    }


if __name__ == "__main__":
    max_order = 6
    reg = 1
    x = np.array([-10, -8, -3, -1, 2, 7])
    y = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09])
    xt = np.array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9])
    yt = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05])

    no_reg = run_bias_variance_tradeoff(x, y, xt, yt, max_order=max_order, reg=None)
    print("====== No Regularization =======")
    print("Training MSE: ", no_reg["train_mse"])
    print("Test MSE: ", no_reg["test_mse"])

    with_reg = run_bias_variance_tradeoff(x, y, xt, yt, max_order=max_order, reg=reg)
    print("====== Regularization =======")
    print("Training Loss", with_reg["train_loss"])
    print("Training MSE: ", with_reg["train_mse"])
    print("Test MSE: ", with_reg["test_mse"])
