import numpy as np


def cov(list1, list2):
    """Return the population covariance between two 1D sequences.

    Step-by-step:
    1. Flatten both inputs to 1D numpy arrays and verify equal length.
    2. Compute sample means and return the mean of (x - mean_x)*(y - mean_y).
    """
    x = np.asarray(list1, dtype=float).ravel()
    y = np.asarray(list2, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("cov() requires inputs of the same length")

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return float(np.mean((x - x_mean) * (y - y_mean)))


def standard_dev(data_list):
    """Return the population standard deviation of a 1D sequence.

    Step-by-step:
    1. Flatten input to 1D array.
    2. Compute sqrt(mean((x - mean(x))^2)) and return as float.
    """
    data = np.asarray(data_list, dtype=float).ravel()
    return float(np.sqrt(np.mean((data - np.mean(data)) ** 2)))


def pearson_correlation(list1, list2):
    """Return Pearson's r for two 1D sequences.

    Step-by-step:
    1. Compute the population standard deviations of both inputs.
    2. If either std is zero the correlation is undefined (raise error).
    3. Return covariance / (std1 * std2).
    """
    std1 = standard_dev(list1)
    std2 = standard_dev(list2)

    if std1 == 0 or std2 == 0:
        raise ValueError("pearson_correlation() is undefined for constant inputs")

    return cov(list1, list2) / (std1 * std2)


if __name__ == "__main__":
    list1 = [0.0838, -0.4092, -0.3025, 1.4261, 0.4658]
    target_list = [0.8206, 1.0639, 0.6895, -0.0252, 0.995]

    print(f"Covariance: {cov(list1, target_list):.4f}")
    print(f"Std Dev (List 1): {standard_dev(list1):.4f}")
    print(f"Std Dev (Target): {standard_dev(target_list):.4f}")
    print(f"Pearson Correlation (r): {pearson_correlation(list1, target_list):.4f}")
