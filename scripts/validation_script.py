import sys
import numpy as np
sys.path.insert(0, './Tutorials/python/python_scripts/EE2211-Ultimate-Cheatsheet')

from matrix_functions import determinant, inverse, are_linearly_independent, solveLE, matrix_rank_value
from statistical_functions import cov, standard_dev
from pearson_correlation import pearson_correlation
from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from RidgeRegression import ridge_regression
from RidgePolynomialRegression import ridge_poly_regression
from gradient_descent_one_variable import gradient_descent_1d
from gradient_descent_two_variable import gradient_descent_2d
from k_means_clustering import run_kmeans

np.random.seed(42)

print("\n=== MATRIX FUNCTIONS ===")
# Determinant
X1 = np.array([[2, 3], [5, 6]])
print("--- Determinant ---")
print("Original Input:", X1.tolist())
print("Custom det  :", determinant(X1))
print("Numpy  det  :", np.linalg.det(X1))

X1_new = np.random.rand(3,3)
print("\nGenerated Input:", X1_new.tolist())
print("Custom det  :", determinant(X1_new))
print("Numpy  det  :", np.linalg.det(X1_new))

print("\n--- Inverse ---")
print("Original Input:", X1.tolist())
print("Custom inv  :\n", inverse(X1))
print("Numpy  inv  :\n", np.linalg.inv(X1))

# Linear Independence
V = [[1, 1], [1, 1]]
print("\n--- Linear Independence ---")
print("Original Input:", V)
ind, rk, n = are_linearly_independent(V)
print("Custom:", ind, "(Rank", rk, ") Expected: False (Rank", np.linalg.matrix_rank(V), ")")

V_new = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
ind_new, rk_new, _ = are_linearly_independent(V_new)
print("\nGenerated Input:", V_new)
print("Custom:", ind_new, "(Rank", rk_new, ") Expected: True (Rank", np.linalg.matrix_rank(V_new), ")")


print("\n=== STATISTICAL FUNCTIONS ===")
f1 = [0.0838, -0.4092, -0.3025, 1.4261, 0.4658]
t1 = [0.8206, 1.0639, 0.6895, -0.0252, 0.995]
print("--- Covariance ---")
print("Custom cov (pop):", cov(f1, t1))
print("Numpy  cov (pop):", np.cov(f1, t1, bias=True)[0, 1])

f2 = np.random.rand(5).tolist()
t2 = np.random.rand(5).tolist()
print("\nGenerated Input - f:", [round(x,4) for x in f2], "t:", [round(x,4) for x in t2])
print("Custom cov (pop):", cov(f2, t2))
print("Numpy  cov (pop):", np.cov(f2, t2, bias=True)[0, 1])

print("\n--- Pearson Correlation ---")
print("Custom r:", pearson_correlation(f1, t1))
print("Numpy  r:", np.corrcoef(f1, t1)[0, 1])

print("\nGenerated r:", pearson_correlation(f2, t2))
print("Numpy     r:", np.corrcoef(f2, t2)[0, 1])


print("\n=== K-MEANS CLUSTERING ===")
X_k = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]])
C_k = np.array([[0, 0], [3, 0]])
res = run_kmeans(X_k, C_k, 10)
print("Original Input Data:\n", X_k.tolist())
print("Original Centroids:\n", C_k.tolist())
print("Custom converged centroids:\n", res["converged"]["centroids"])

X_k2 = np.random.rand(6, 2)
C_k2 = X_k2[:2]
res2 = run_kmeans(X_k2, C_k2, 10)
print("\nGenerated Input Data:\n", X_k2.tolist())
print("Custom converged centroids:\n", res2["converged"]["centroids"])

print("\n=== LINEAR REGRESSION ===")
# Overdetermined system
X_lr = np.array([[1,2],[1,6],[1,0],[1,5],[1,7]])
Y_lr = np.array([[1],[2],[3],[4],[5]])
X_test_lr = np.array([[1,3]])

system, w, sse, mse, pred = linear_regression(X_lr, Y_lr, X_test_lr)
w_expected, _, _, _ = np.linalg.lstsq(X_lr, Y_lr, rcond=None)
print("Custom Linear Regression Weights:\n", w)
print("Numpy Expected Weights:\n", w_expected)

print("\n=== RIDGE REGRESSION ===")
lamda = 0.5
X_ridge = np.array([[1,2],[1,6],[1,0],[1,5],[1,7]])
Y_ridge = np.array([[1],[2],[3],[4],[5]])
X_test_ridge = np.array([[1,3]])

# Call the function (it prints the output)
print("--- Custom Ridge Regression Output ---")
ridge_regression(X_ridge, Y_ridge, lamda, X_test_ridge)

N = X_ridge.shape[1]
# exact Ridge weights: (X^T X + lambda I)^-1 X^T Y
w_ridge_expected = np.linalg.inv(X_ridge.T @ X_ridge + lamda * np.eye(N)) @ X_ridge.T @ Y_ridge
print("--- Expected Analytical Ridge Weights ---")
print(w_ridge_expected)

print("\n=== POLYNOMIAL REGRESSION ===")
# Polynomial order 2
X_poly = np.array([[1], [2], [3], [4]])
Y_poly = np.array([[1],[4],[9],[16]])
X_test_poly = np.array([[5]])

print("--- Custom Polynomial Regression Output ---")
polynomial_regression(X_poly, Y_poly, 2, X_test_poly)

print("\n--- Expected Analytical Polynomial Weights ---")
# Exact: fitting y = ax^2 + bx + c for points (1,1), (2,4), (3,9), (4,16)
# Expected w = [0, 0, 1] for x^0, x^1, x^2
from sklearn.preprocessing import PolynomialFeatures
P_exp = PolynomialFeatures(2).fit_transform(X_poly)
w_poly_exp, _, _, _ = np.linalg.lstsq(P_exp, Y_poly, rcond=None)
print(w_poly_exp)


print("\n=== RIDGE POLYNOMIAL REGRESSION ===")
lamda_poly = 0.1
print("--- Custom Ridge Polynomial Output ---")
ridge_poly_regression(X_poly, Y_poly, lamda_poly, 2, "auto", X_test_poly)

print("\n--- Expected Analytical Ridge Poly Weights ---")
P_ridge_exp = PolynomialFeatures(2).fit_transform(X_poly)
N_poly = P_ridge_exp.shape[1]
w_ridge_poly_exp = np.linalg.inv(P_ridge_exp.T @ P_ridge_exp + lamda_poly * np.eye(N_poly)) @ P_ridge_exp.T @ Y_poly
print(w_ridge_poly_exp)

print("\n=== GRADIENT DESCENT 1D ===")
# f(x) = (x-3)^2, min is at x=3
f_1d = lambda x: (x-3)**2
df_1d = lambda x: 2*(x-3)
fx, hist, _, _ = gradient_descent_1d(f_1d, df_1d, start_value=0.0, learning_rate=0.1, iterations=50)
print(f"Custom GD 1D final x: {fx:.6f}")
print("Expected x        : 3.000000")

print("\n=== GRADIENT DESCENT 2D ===")
# f(x,y) = (x-2)^2 + (y+1)^2, min is at (2, -1)
f_2d = lambda x,y: (x-2)**2 + (y+1)**2
df_2d = lambda x,y: (2*(x-2), 2*(y+1))
final_x_2d, final_y_2d, hist_2d, _, _ = gradient_descent_2d(f_2d, df_2d, start_x=0.0, start_y=0.0, learning_rate=0.1, iterations=50)
print(f"Custom GD 2D final pos: ({final_x_2d:.6f}, {final_y_2d:.6f})")
print("Expected pos        : (2.000000, -1.000000)")


print("\n=== VERIFICATION COMPLETE ===")
