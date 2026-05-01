from statistical_functions import pearson_correlation as _pearson_correlation


def pearson_correlation(X, Y):
    """Backward-compatible wrapper for the Pearson correlation objective.

    Step-by-step:
    1. Forward the call to `statistical_functions.pearson_correlation`.
    2. Return the numeric Pearson r value to the caller.
    """
    return _pearson_correlation(X, Y)


if __name__ == "__main__":
    data = [0.3510, 2.1812, 0.2415, -0.1096, 0.1544]
    target = [0.2758, 1.4392, -0.4611, 0.6154, 1.0006]
    print(f"Pearson's r: {pearson_correlation(data, target):.4f}")
