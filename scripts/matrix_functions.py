import numpy as np
from numpy.linalg import inv, matrix_rank


def are_linearly_independent(vectors):
    """Check whether stacked row vectors are linearly independent.

    Step-by-step:
    1. Stack input `vectors` into a 2D numpy array (rows are vectors).
    2. Compute the matrix rank using `numpy.linalg.matrix_rank`.
    3. Compare rank to the number of row vectors; if equal they are independent.
    4. Return a tuple (is_independent, rank, num_vectors).
    """
    matrix = np.asarray(vectors, dtype=float)
    rank = matrix_rank(matrix)
    num_vectors = matrix.shape[0]
    return bool(rank == num_vectors), int(rank), int(num_vectors)


def matrix_rank_value(matrix):
    # Step-by-step:
    # 1. Convert input to numpy array and compute the matrix rank.
    # 2. Return the rank as an integer.
    return int(matrix_rank(np.asarray(matrix, dtype=float)))


def determinant(matrix):
    # Step-by-step:
    # 1. Convert to numpy array and check squareness.
    # 2. Use `numpy.linalg.det` to compute and return determinant as float.
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("determinant() requires a square matrix")
    return float(np.linalg.det(matrix))


def inverse(matrix):
    # Step-by-step:
    # 1. Convert to numpy array and check squareness.
    # 2. Use `numpy.linalg.inv` to compute and return the inverse.
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse() requires a square matrix")
    return inv(matrix)


def left_right_invertible(matrix):
    # Step-by-step:
    # 1. Convert input to numpy array and determine its shape (m rows, n cols).
    # 2. Compute rank. A matrix is left-invertible when rank == n and m >= n.
    # 3. It is right-invertible when rank == m and n >= m.
    # 4. Return a dict describing left/right invertibility, rank and shape.
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
    # Step-by-step:
    # 1. Convert X and y to arrays and ensure y is a column vector.
    # 2. Form the augmented matrix [X|y] and compute ranks of X and [X|y].
    # 3. Compare ranks to determine the rank condition RC:
    #    RC=1 => unique solution (rankX == rankX_ == d),
    #    RC=2 => inconsistent (rankX < rankX_),
    #    RC=3 => infinitely many solutions (rankX == rankX_ < d).
    # 4. Return (RC, rankX, rankX_).
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
    # Step-by-step:
    # 1. Determine the shape relation of X (even/over/under) using det_checker.
    # 2. Dispatch to the corresponding solver (evenSolver/overSolver/underSolver).
    # 3. Print a human-readable result and return the solution `w` (or None if not available).
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
