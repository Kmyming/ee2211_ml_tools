import numpy as np
from sklearn.metrics import mean_squared_error


def _as_1d_array(values, name):
	"""Convert input to 1D numpy array with validation."""
	array = np.asarray(values, dtype=float)
	if array.ndim == 0:
		array = array.reshape(1)
	elif array.ndim == 2 and array.shape[1] == 1:
		array = array.ravel()
	elif array.ndim > 1:
		raise ValueError(f"{name} must be a 1D array or a column vector.")
	return array.ravel()


def mse(y):
	"""
	Mean Squared Error impurity measure.
	
	For regression: MSE = mean((y - y_mean)^2)
	
	Args:
		y: 1D array of continuous target values.
	
	Returns:
		float: MSE value.
	"""
	y = _as_1d_array(y, "y")
	if y.size == 0:
		return 0.0
	return mean_squared_error(y, np.full(y.shape, np.mean(y)))


def entropy(y):
	"""
	Shannon Entropy impurity measure (classification).
	
	Entropy = -sum(p_i * log2(p_i)) where p_i is the fraction of class i.
	
	Args:
		y: 1D array of discrete class labels (treated as categories).
	
	Returns:
		float: Entropy value (range 0 to log2(num_classes)).
	"""
	y = _as_1d_array(y, "y")
	if y.size == 0:
		return 0.0
	
	unique_classes, counts = np.unique(y, return_counts=True)
	proportions = counts / len(y)
	
	# Filter out zero proportions to avoid log(0)
	proportions = proportions[proportions > 0]
	
	entropy_val = -np.sum(proportions * np.log2(proportions))
	return float(entropy_val)


def gini(y):
	"""
	Gini Impurity measure (classification).
	
	Gini = 1 - sum(p_i^2) where p_i is the fraction of class i.
	
	Args:
		y: 1D array of discrete class labels (treated as categories).
	
	Returns:
		float: Gini impurity value (range 0 to 1 - 1/num_classes).
	"""
	y = _as_1d_array(y, "y")
	if y.size == 0:
		return 0.0
	
	unique_classes, counts = np.unique(y, return_counts=True)
	proportions = counts / len(y)
	gini_val = 1.0 - np.sum(proportions ** 2)
	return float(gini_val)


def misclassification_rate(y):
	"""
	Misclassification Rate impurity measure (classification).
	
	Misclassification Rate = 1 - max(p_i) where p_i is the fraction of class i.
	
	Args:
		y: 1D array of discrete class labels (treated as categories).
	
	Returns:
		float: Misclassification rate (range 0 to 1 - 1/num_classes).
	"""
	y = _as_1d_array(y, "y")
	if y.size == 0:
		return 0.0
	
	unique_classes, counts = np.unique(y, return_counts=True)
	proportions = counts / len(y)
	misclass_rate = 1.0 - np.max(proportions)
	return float(misclass_rate)


def calculate_impurity(y, measure_type="mse"):
	"""
	Calculate impurity for a given measure type.
	
	Args:
		y: 1D array of target values.
		measure_type: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		float: Impurity value.
	
	Raises:
		ValueError: If measure_type is not recognized.
	"""
	measure_type = measure_type.lower().strip()
	
	if measure_type == "mse":
		return mse(y)
	elif measure_type == "entropy":
		return entropy(y)
	elif measure_type == "gini":
		return gini(y)
	elif measure_type == "misclassification":
		return misclassification_rate(y)
	else:
		raise ValueError(f"Unknown impurity measure: '{measure_type}'. "
						 f"Must be one of: 'mse', 'entropy', 'gini', 'misclassification'.")


if __name__ == "__main__":
	# Quick test
	y_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
	y_classification = np.array([0, 0, 1, 1, 2])
	
	print("=== Regression Example ===")
	print(f"Data: {y_regression}")
	print(f"MSE: {calculate_impurity(y_regression, 'mse'):.4f}")
	
	print("\n=== Classification Example ===")
	print(f"Data: {y_classification}")
	print(f"Entropy: {calculate_impurity(y_classification, 'entropy'):.4f}")
	print(f"Gini: {calculate_impurity(y_classification, 'gini'):.4f}")
	print(f"Misclassification: {calculate_impurity(y_classification, 'misclassification'):.4f}")
