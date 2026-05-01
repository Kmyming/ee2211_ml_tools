from unittest import result

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from decision_tree_impurity import overall_impurity, tree_impurity
from decision_tree import tree_impurity_summary, tree_overall_impurity
from regression_tree import tree_impurity_summary_auto, tree_overall_impurity_auto
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
from bias_variance_tradeoff import run_bias_variance_tradeoff
from activation_functions import relu, sigmoid


# Change this value to choose exactly one example to run.
SELECTED_EXAMPLE = "regression_tree"


def run_matrix_linear_independence_example():
	# Input: a list of row vectors.
	# Output: a tuple of (is_independent, rank, number_of_vectors).
	vectors = [[1, 2], [1, 1]]
	independent, rank, num_vectors = are_linearly_independent(vectors)
	print("Linearly Independent" if independent else "Linearly Dependent")
	print(f"Rank: {rank} / Number of vectors: {num_vectors}")


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
	Y = np.array([[1], [2], [3], [2]])
	onehot = OneHotEncoder(sparse_output=False)
	Y_onehot = onehot.fit_transform(Y)
	X_test = np.array([[2, 1]])
	X_fitted = np.hstack((np.ones((len(X), 1)), X))
	X_test_fitted = np.hstack((np.ones((len(X_test), 1)), X_test))
	onehot_linearclassification(X_fitted, Y_onehot, X_test_fitted)


def run_onehot_polynomial_regression_example():
	# Input: raw features X, target Y, polynomial order, and raw test features X_test.
	# Output: polynomial class scores and the predicted class index.
	X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]])
	Y = np.array([[2], [3.1], [3.5], [4]])
	onehot = OneHotEncoder(sparse_output=False)
	Y_onehot = onehot.fit_transform(Y)
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


def run_optimization_gradient_descent_1d_example():
	# Input: default 1-variable objective, starting value, learning rate, and iteration count.
	# Output: final x iteration values and per-iteration history.
	func = lambda x: np.cos(x ** 2) ** 2
	grad_func = lambda x: -2 * x * np.sin(2 * x ** 2)
	start_value = 1
	learning_rate = 0.1
	iterations = 4
	final_x, history, _, _ = gradient_descent_1d(func, grad_func, start_value, learning_rate, iterations)
	print("Function: f(x) = cos(x^2)^2")
	print("Derivative: df/dx = -2*x*sin(2*x^2)")
	print("-" * 30)
	for step in history:
		print(f"Iteration {step['iteration']}:")
		print(f"  x        = {step['x']:.5f}")
		print(f"  f(x)     = {step['f_x']:.5f}")
		print(f"  gradient = {step['gradient']:.5f}")
		print("-" * 30)
	print("Final x:", final_x)


def run_optimization_gradient_descent_2d_example():
	# Input: default 2-variable objective, starting point, learning rate, and iteration count.
	# Output: final x,y iteration values  and per-iteration history.
	func = lambda x, y: x ** 2
	grad_func = lambda x, y: (2.0 * x, 0.0)
	start_x=3.0
	start_y=2.0
	learning_rate=0.2
	iterations=5
	x_value, y_value, history, _, _ = gradient_descent_2d(func, grad_func, start_x, start_y, learning_rate, iterations)
	print("Function: f(x,y) = x^2")
	print("df/dx = 2*x")
	print("df/dy = 0")
	print("-" * 30)
	for step in history:
		print(f"Iteration {step['iteration']}:")
		print(f"  Gradient: ({step['gradient_x']:.4f}, {step['gradient_y']:.4f})")
		print(f"  New Position: (x={step['x']:.4f}, y={step['y']:.4f})")
		print(f"  Function Value: {step['f_xy']:.4f}")
		print("-" * 15)
	print("Final position:", (x_value, y_value))


def run_optimization_bias_variance_tradeoff_example():
	# Input: x, y, xt, yt sequences, max polynomial order, and optional regularization.
	# Output: training/test MSE curves, and training loss when regularization is enabled.
	x = np.array([-10, -8, -3, -1, 2, 7])
	y = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09])
	xt = np.array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9])
	yt = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05])
	max_order = 6
	reg = 1

	no_reg = run_bias_variance_tradeoff(x, y, xt, yt, max_order=max_order, reg=None)
	print("====== No Regularization =======")
	print("Training MSE:", no_reg["train_mse"])
	print("Test MSE:", no_reg["test_mse"])

	with_reg = run_bias_variance_tradeoff(x, y, xt, yt, max_order=max_order, reg=reg)
	print("====== Regularization =======")
	print("Training Loss", with_reg["train_loss"])
	print("Training MSE:", with_reg["train_mse"])
	print("Test MSE:", with_reg["test_mse"])


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


def run_tree_decision_example():
	# Input: 1D feature values, target values, and threshold path definitions.
	# Output: overall leaf-prediction MSE and a depth-by-depth split summary.
	X = np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9])
	y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5])
	thresholds = [("", 5)]
	
	# Example 1: MSE-based splitting (default)
	print("===== Decision Tree with MSE-based splitting (default) =====")
	tree_overall_impurity(X, y, thresholds, impurity_measure="mse")
	tree_impurity_summary(X, y, thresholds, max_depth=2, impurity_measure="mse")
	
	# Example 2: Alternative - Gini-based splitting
	# Uncomment the lines below to compare with Gini impurity measure
	# print("\n===== Decision Tree with Gini-based splitting (alternative) =====")
	# tree_overall_impurity(X, y, thresholds, impurity_measure="gini")
	# tree_impurity_summary(X, y, thresholds, max_depth=2, impurity_measure="gini")


def run_tree_regression_example():
	# Input: 1D feature values, target values, and a max depth.
	# Output: automatically selected thresholds, leaf predictions, and depth-by-depth impurity summary.
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])
	
	# Example 1: Automatic splits with MSE (default)
	print("===== Regression Tree with automatic MSE-based splits (default) =====")
	tree_overall_impurity_auto(X, y, max_depth=1, impurity_measure="mse")
	tree_impurity_summary_auto(X, y, max_depth=1, impurity_measure="mse")
	
	# Example 2: Alternative - Automatic splits with Gini
	# Uncomment the lines below to see how Gini-based splitting differs from MSE
	# print("\n===== Regression Tree with automatic Gini-based splits (alternative) =====")
	# tree_overall_impurity_auto(X, y, max_depth=1, impurity_measure="gini")
	# tree_impurity_summary_auto(X, y, max_depth=1, impurity_measure="gini")



def run_kmeans_example():
	# Input: 2D data points, initial centroids, and max_iter.
	# Output: centroids and labels after limited iterations and after convergence.
	X = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]])
	init_centroids = np.array([[0, 0], [3, 0]])
	max_iter= 2
	result = run_kmeans(X, init_centroids, max_iter)
	print(f"=== After {max_iter} Iteration(s) ===")
	print("Centroids:\n", result["partial"]["centroids"])
	print("Predictions:", result["partial"]["labels"])
	print("\n=== After Convergence ===")
	print("Centroids:\n", result["converged"]["centroids"])
	print("Predictions:", result["converged"]["labels"])


def run_activation_functions_example():
	# Input: a 1D array of values.
	# Output: ReLU and sigmoid activations applied elementwise.
	arr = np.array([-2, -1, 0, 1, 2])
	relu_result = relu(arr)
	sigmoid_result = sigmoid(arr)
	print("Input array:", arr)
	print("ReLU output:", relu_result)
	print("Sigmoid output:", np.round(sigmoid_result, 6))




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
		"bias_variance_tradeoff": run_optimization_bias_variance_tradeoff_example,
		"gradient_descent_1d": run_optimization_gradient_descent_1d_example,
		"gradient_descent_2d": run_optimization_gradient_descent_2d_example,
	},
	"tree_examples": {
		"impurity": run_tree_impurity_example,
		"decision_tree": run_tree_decision_example,
		"regression_tree": run_tree_regression_example,
	},
	"clustering_examples": {
		"kmeans": run_kmeans_example,
		"activation_functions": run_activation_functions_example,
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


def run_selected_example(example):
	matches = []
	for category, category_examples in EXAMPLE_REGISTRY.items():
		if example in category_examples:
			matches.append((category, category_examples[example]))

	if not matches:
		raise ValueError(f"Unknown example '{example}'. Available examples: {[name for category in EXAMPLE_REGISTRY.values() for name in category]}")
	if len(matches) > 1:
		raise ValueError(f"Example '{example}' exists in multiple categories. Rename one of them to keep example-only selection unambiguous.")

	category, example_func = matches[0]
	print(f"Running {category} -> {example}\n")
	example_func()


if __name__ == "__main__":
	run_selected_example(SELECTED_EXAMPLE)
