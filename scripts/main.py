import numpy as np

from decision_tree_impurity import overall_impurity, tree_impurity
from decision_tree_mse import tree_MSE_summary, tree_overall_MSE
from gradient_descent_one_variable import gradient_descent_1d
from gradient_descent_two_variable import gradient_descent_2d
from k_means_clustering import run_kmeans
from matrix_functions import (
	RC_checker,
	are_linearly_independent,
	determinant,
	inverse,
	left_right_invertible,
	matrix_rank_value,
	solveLE,
)
from pearson_correlation import pearson_correlation
from statistical_functions import cov, standard_dev
from LinearRegression import linear_regression
from OneHotLinearClassification import onehot_linearclassification
from OneHotPolynomialRegression import polynomial_regression_onehot
from PolynomialRegression import polynomial_regression
from RidgeOneHot import ridge_linear_regression_onehot
from RidgePolyOneHot import ridge_poly_regression_onehot
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression


# Change these two values to choose exactly one example to run.
SELECTED_CATEGORY = "regression_examples"
SELECTED_EXAMPLE = "linear_regression"


def run_matrix_linear_independence_example():
	# Input: a list of row vectors.
	# Output: a tuple of (is_independent, rank, number_of_vectors).
	vectors = [[1, 1], [1, 1]]
	print("Matrix: linear independence")
	print(are_linearly_independent(vectors))


def run_matrix_rank_example():
	# Input: a matrix.
	# Output: the matrix rank as an integer.
	X = np.array([[1, 1], [1, 1]])
	print("Matrix rank:", matrix_rank_value(X))


def run_matrix_determinant_example():
	# Input: a square matrix.
	# Output: the determinant as a float.
	X = np.array([[2, 3], [5, 6]])
	print("Determinant:", determinant(X))


def run_matrix_inverse_example():
	# Input: an invertible square matrix.
	# Output: the inverse matrix.
	X = np.array([[3, 2], [2, 4]])
	print("Inverse:\n", inverse(X))


def run_matrix_invertibility_example():
	# Input: any matrix.
	# Output: dictionary describing left/right invertibility and rank.
	X = np.array([[1, 2], [0, 1], [2, 3]])
	print("Invertibility info:", left_right_invertible(X))


def run_matrix_rc_example():
	# Input: matrix X and target y for a linear system.
	# Output: RC classification and the ranks of X and [X|y].
	X = np.array([[1, 2], [3, 4]])
	y = np.array([[5], [11]])
	print("Rank condition:", RC_checker(X, y))


def run_matrix_solve_le_example():
	# Input: matrix X and target y for Xw = y.
	# Output: the weight vector w if a solution exists.
	X = np.array([[1, 2], [3, 4]])
	y = np.array([[5], [11]])
	print("Linear equation solution:\n", solveLE(X, y))


def run_statistics_cov_example():
	# Input: two same-length 1D sequences.
	# Output: population covariance.
	feature = [0.0838, -0.4092, -0.3025, 1.4261, 0.4658]
	target = [0.8206, 1.0639, 0.6895, -0.0252, 0.995]
	print("Covariance:", cov(feature, target))


def run_statistics_std_example():
	# Input: one 1D sequence.
	# Output: population standard deviation.
	feature = [0.0838, -0.4092, -0.3025, 1.4261, 0.4658]
	print("Standard deviation:", standard_dev(feature))


def run_statistics_pearson_example():
	# Input: two same-length 1D sequences.
	# Output: Pearson correlation coefficient r.
	feature = [0.0838, -0.4092, -0.3025, 1.4261, 0.4658]
	target = [0.8206, 1.0639, 0.6895, -0.0252, 0.995]
	print("Pearson's r:", pearson_correlation(feature, target))


def run_optimization_gradient_descent_1d_example():
	# Input: default 1-variable objective, starting value, learning rate, and iteration count.
	# Output: final x value and per-iteration history.
	final_x, history, _, _ = gradient_descent_1d()
	print("Gradient descent (1 variable) final x:", final_x)
	print("Gradient descent (1 variable) history:", history)


def run_optimization_gradient_descent_2d_example():
	# Input: default 2-variable objective, starting point, learning rate, and iteration count.
	# Output: final (x, y) and per-iteration history.
	final_x, final_y, history, _, _ = gradient_descent_2d()
	print("Gradient descent (2 variables) final position:", (final_x, final_y))
	print("Gradient descent (2 variables) history:", history)


def run_tree_impurity_example():
	# Input: class counts at each node for each tree depth.
	# Output: node-level and depth-level Gini, entropy, and misclassification impurity.
	layers = [
		[[8, 5, 5]],
		[[6, 4, 0], [2, 1, 5]],
		[],
	]
	print("Tree impurity summaries:", tree_impurity(layers))
	print("Overall impurity summary:", overall_impurity(layers, selected_depths=[1]))


def run_tree_mse_example():
	# Input: 1D feature values, target values, and threshold path definitions.
	# Output: overall leaf-prediction MSE and a depth-by-depth split summary.
	X = np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9])
	y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5])
	thresholds = [("", 5)]
	print("Tree overall MSE:", tree_overall_MSE(X, y, thresholds))
	print("Tree MSE summary:", tree_MSE_summary(X, y, thresholds, max_depth=1))


def run_kmeans_example():
	# Input: 2D data points, initial centroids, and max_iter.
	# Output: centroids and labels after limited iterations and after convergence.
	X = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]])
	init_centroids = np.array([[0, 0], [3, 0]])
	result = run_kmeans(X, init_centroids, max_iter=2)
	print("KMeans partial centroids:\n", result["partial"]["centroids"])
	print("KMeans partial labels:", result["partial"]["labels"])
	print("KMeans converged centroids:\n", result["converged"]["centroids"])
	print("KMeans converged labels:", result["converged"]["labels"])


def run_linear_regression_example():
	# Input: X is the raw training feature matrix, Y is the target vector, X_test is the raw test matrix.
	# Output: fitted weights, training error, and predicted test output.
	X = np.array([[1, 2], [0, 6], [1, 0], [0, 5], [1, 7]])
	Y = np.array([[1], [2], [3], [4], [5]])
	X_test = np.array([[1, 3]])
	X_fitted = np.hstack((np.ones((len(X), 1)), X))
	X_test_fitted = np.hstack((np.ones((len(X_test), 1)), X_test))
	linear_regression(X_fitted, Y, X_test_fitted)


def run_polynomial_regression_example():
	# Input: raw features X, target Y, polynomial order, and raw test features X_test.
	# Output: polynomial feature matrix, fitted weights, training error, and test prediction.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	X_test = np.array([[1, -2]])
	polynomial_regression(X, Y, 2, X_test)


def run_ridge_regression_example():
	# Input: fitted feature matrix X, target Y, ridge lambda, and fitted test matrix X_test.
	# Output: ridge weights, training error, and test prediction.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	X_test = np.array([[1, -2]])
	X_fitted = np.hstack((np.ones((len(X), 1)), X))
	X_test_fitted = np.hstack((np.ones((len(X_test), 1)), X_test))
	ridge_regression(X_fitted, Y, 0.1, X_test_fitted)


def run_ridge_polynomial_regression_example():
	# Input: raw features X, target Y, ridge lambda, polynomial order, form selector, and raw test features X_test.
	# Output: ridge polynomial weights, training error, and test prediction.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	X_test = np.array([[1, -2]])
	ridge_poly_regression(X, Y, 0.1, 2, "auto", X_test)


def run_onehot_linear_classification_example():
	# Input: fitted feature matrix X, class targets Y, and fitted test matrix X_test.
	# Output: estimated class scores and the predicted class index.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	X_test = np.array([[1, -2]])
	X_fitted = np.hstack((np.ones((len(X), 1)), X))
	X_test_fitted = np.hstack((np.ones((len(X_test), 1)), X_test))
	onehot_linearclassification(X_fitted, Y, X_test_fitted)


def run_onehot_polynomial_regression_example():
	# Input: raw features X, target Y, polynomial order, and raw test features X_test.
	# Output: polynomial class scores and the predicted class index.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	X_test = np.array([[1, -2]])
	polynomial_regression_onehot(X, Y, 2, X_test)


def run_ridge_onehot_linear_classification_example():
	# Input: fitted feature matrix X, class targets Y, ridge lambda, and fitted test matrix X_test.
	# Output: ridge class scores and predicted class index.
	X = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 3], [3, 1, 4], [-1, 2, 1]])
	Y = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
	X_test = np.array([[1, 1, 2]])
	X_fitted = np.hstack((np.ones((len(X), 1)), X))
	X_test_fitted = np.hstack((np.ones((len(X_test), 1)), X_test))
	ridge_linear_regression_onehot(X_fitted, Y, 0.01, 2, "auto", X_test_fitted)


def run_ridge_onehot_polynomial_regression_example():
	# Input: raw features X, class targets Y, ridge lambda, polynomial order, and raw test features X_test.
	# Output: ridge polynomial class scores and predicted class index.
	X = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 3], [3, 1, 4], [-1, 2, 1]])
	Y = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
	X_test = np.array([[1, 1, 2]])
	ridge_poly_regression_onehot(X, Y, 0.01, 2, "auto", X_test)


EXAMPLE_REGISTRY = {
	"matrix_examples": {
		"linear_independence": run_matrix_linear_independence_example,
		"rank": run_matrix_rank_example,
		"determinant": run_matrix_determinant_example,
		"inverse": run_matrix_inverse_example,
		"invertibility": run_matrix_invertibility_example,
		"rank_condition": run_matrix_rc_example,
		"solve_le": run_matrix_solve_le_example,
	},
	"statistics_examples": {
		"covariance": run_statistics_cov_example,
		"standard_deviation": run_statistics_std_example,
		"pearson_r": run_statistics_pearson_example,
	},
	"optimization_examples": {
		"gradient_descent_1d": run_optimization_gradient_descent_1d_example,
		"gradient_descent_2d": run_optimization_gradient_descent_2d_example,
	},
	"tree_examples": {
		"impurity": run_tree_impurity_example,
		"mse": run_tree_mse_example,
	},
	"clustering_examples": {
		"kmeans": run_kmeans_example,
	},
	"regression_examples": {
		"linear_regression": run_linear_regression_example,
		"polynomial_regression": run_polynomial_regression_example,
		"ridge_regression": run_ridge_regression_example,
		"ridge_polynomial_regression": run_ridge_polynomial_regression_example,
		"onehot_linear_classification": run_onehot_linear_classification_example,
		"onehot_polynomial_regression": run_onehot_polynomial_regression_example,
		"ridge_onehot_linear_classification": run_ridge_onehot_linear_classification_example,
		"ridge_onehot_polynomial_regression": run_ridge_onehot_polynomial_regression_example,
	},
}


def run_selected_example(category, example):
	if category not in EXAMPLE_REGISTRY:
		raise ValueError(f"Unknown category '{category}'. Available: {list(EXAMPLE_REGISTRY)}")

	category_examples = EXAMPLE_REGISTRY[category]
	if example not in category_examples:
		raise ValueError(f"Unknown example '{example}' for category '{category}'. Available: {list(category_examples)}")

	print(f"Running {category} -> {example}\n")
	category_examples[example]()


if __name__ == "__main__":
	run_selected_example(SELECTED_CATEGORY, SELECTED_EXAMPLE)
