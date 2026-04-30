import numpy as np
from numpy.linalg import inv, matrix_rank


def are_linearly_independent(vectors):
    """Check whether stacked row vectors are linearly independent."""
    matrix = np.asarray(vectors, dtype=float)
    rank = matrix_rank(matrix)
    num_vectors = matrix.shape[0]
    return bool(rank == num_vectors), int(rank), int(num_vectors)


def matrix_rank_value(matrix):
    return int(matrix_rank(np.asarray(matrix, dtype=float)))


def determinant(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("determinant() requires a square matrix")
    return float(np.linalg.det(matrix))


def inverse(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse() requires a square matrix")
    return inv(matrix)


def left_right_invertible(matrix):
    matrix = np.asarray(matrix, dtype=float)
    m, n = matrix.shape
    rank = matrix_rank(matrix)
    return {
        "left_invertible": bool((m >= n) and (rank == n)),
        "right_invertible": bool((n >= m) and (rank == m)),
        "rank": int(rank),
        "shape": (m, n),
    }


def det_checker(X):
    X = np.asarray(X, dtype=float)
    m, d = X.shape
    if m == d:
        return "even"
    if m > d:
        return "over"
    return "under"


def RC_checker(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    X_ = np.append(X, y, axis=1)
    rankX = matrix_rank(X)
    rankX_ = matrix_rank(X_)
    d = X.shape[1]

    if rankX == rankX_:
        if rankX == d:
            RC = 1
        else:
            RC = 3
    else:
        RC = 2
    return RC, int(rankX), int(rankX_)


def evenSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    w = None
    if RC == 1:
        w = inv(X) @ y
        ans = "unique."
    elif RC == 2:
        ans = "No solution."
    else:
        ans = "Infinitely many solutions."
    return w, ans


def overSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    w = None
    if RC == 1:
        w = inv(X.T @ X) @ X.T @ y
        ans = "Unique."
    elif RC == 3:
        ans = "Infinitely many solutions."
    elif determinant(X.T @ X) != 0:
        w = inv(X.T @ X) @ X.T @ y
        ans = "No exact solution, but least square approximation can be found. Left-inv."
    else:
        ans = "No solution."
    return w, ans


def underSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    w = None
    if RC == 2:
        ans = "No solution."
    elif determinant(X @ X.T) != 0:
        w = X.T @ inv(X @ X.T) @ y
        ans = "No exact solution, but least norm approximation can be found. Right-inv."
    else:
        ans = "Infinitely many solutions."
    return w, ans


def solveLE(X, y):
    detX = det_checker(X)
    if detX == "even":
        w, ans = evenSolver(X, y)
    elif detX == "over":
        w, ans = overSolver(X, y)
    else:
        w, ans = underSolver(X, y)

    print("\n", ans, f"\nw = \n{w}")
    return w


if __name__ == "__main__":
    v1 = [1, 1]
    v2 = [1, 1]
    independent, rank, num_vectors = are_linearly_independent([v1, v2])
    print("Linearly Independent" if independent else "Linearly Dependent")
    print(f"Rank: {rank} / Number of vectors: {num_vectors}")

    X = np.array([[1, 2], [3, 4]])
    print("Rank:", matrix_rank_value(X))
    print("Determinant:", round(determinant(X), 4))
    print("Inverse:\n", inverse(X))
    print(left_right_invertible(np.array([[1, 2], [0, 1], [2, 3]])))
