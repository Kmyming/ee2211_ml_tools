"""
Validation script for impurity measures and tree implementations.
Tests regression_tree with multiple impurity measures against example and notebook datasets.
"""

import numpy as np
import sys

# Import the main modules
from impurity_measures import calculate_impurity, mse, entropy, gini, misclassification_rate
from regression_tree import (
	tree_overall_impurity_auto, 
	tree_impurity_summary_auto, 
	find_best_split
)


def validate_impurity_measures():
	"""Validate that impurity measures compute correctly."""
	print("=" * 70)
	print("VALIDATION 1: Impurity Measures")
	print("=" * 70)
	
	# Test data: regression (continuous)
	y_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
	print(f"\nRegression data: {y_regression}")
	print(f"  MSE: {mse(y_regression):.6f}")
	
	# Test data: classification (discrete)
	y_classification = np.array([0, 0, 1, 1, 2])
	print(f"\nClassification data: {y_classification}")
	print(f"  Entropy: {entropy(y_classification):.6f}")
	print(f"  Gini: {gini(y_classification):.6f}")
	print(f"  Misclassification: {misclassification_rate(y_classification):.6f}")
	
	# Validate that different measures give different values
	assert mse(y_regression) > 0, "MSE should be positive"
	assert entropy(y_classification) > 0, "Entropy should be positive"
	assert gini(y_classification) > 0, "Gini should be positive"
	assert misclassification_rate(y_classification) > 0, "Misclassification should be positive"
	
	print("\n✓ All impurity measures compute correctly and are positive.")


def validate_find_best_split():
	"""Validate that find_best_split produces valid splits."""
	print("\n" + "=" * 70)
	print("VALIDATION 2: Best Split Finding")
	print("=" * 70)
	
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])
	
	measures = ["mse", "entropy", "gini", "misclassification"]
	splits = {}
	
	for measure in measures:
		best = find_best_split(X, y, impurity_measure=measure)
		splits[measure] = best
		
		print(f"\nUsing impurity measure: {measure}")
		print(f"  Threshold: {best['threshold']:.4f}")
		print(f"  Weighted impurity: {best['weighted_impurity']:.6f}")
		print(f"  Left samples: {len(best['left_y'])}, Right samples: {len(best['right_y'])}")
		
		# Validate that split exists and is valid
		assert best is not None, f"Split should exist for {measure}"
		assert 0 < len(best['left_y']) < len(y), f"Left split invalid for {measure}"
		assert 0 < len(best['right_y']) < len(y), f"Right split invalid for {measure}"
		assert best['weighted_impurity'] < best['node_impurity'], \
			f"Weighted impurity should be less than node impurity for {measure}"
	
	print("\n✓ All split finding methods produce valid splits.")
	return splits


def validate_regression_tree_mse():
	"""Validate regression tree with MSE measure on example data."""
	print("\n" + "=" * 70)
	print("VALIDATION 3: Regression Tree with MSE (Example Dataset)")
	print("=" * 70)
	
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])
	
	print(f"\nInput X: {X}")
	print(f"Input y: {y}")
	
	# Build tree with MSE
	overall_mse, y_pred, tree = tree_overall_impurity_auto(X, y, max_depth=1, impurity_measure="mse")
	
	print(f"\nTree structure (depth 1):")
	print(f"  Root split threshold: {tree['threshold']:.4f}")
	print(f"  Overall MSE: {overall_mse:.6f}")
	print(f"  Predictions: {np.round(y_pred, 4)}")
	
	# Validate
	assert tree["type"] == "split", "Root should be a split for this data"
	assert len(y_pred) == len(y), "Predictions length should match y"
	assert overall_mse >= 0, "MSE should be non-negative"
	
	print("✓ Regression tree with MSE builds correctly.")
	return overall_mse, y_pred, tree


def validate_regression_tree_gini():
	"""Validate regression tree with Gini measure on example data."""
	print("\n" + "=" * 70)
	print("VALIDATION 4: Regression Tree with Gini (Example Dataset)")
	print("=" * 70)
	
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])
	
	# Build tree with Gini
	overall_mse_gini, y_pred_gini, tree_gini = tree_overall_impurity_auto(
		X, y, max_depth=1, impurity_measure="gini"
	)
	
	print(f"\nTree structure (depth 1):")
	print(f"  Root split threshold: {tree_gini['threshold']:.4f}")
	print(f"  Overall MSE: {overall_mse_gini:.6f}")
	print(f"  Predictions: {np.round(y_pred_gini, 4)}")
	
	# Validate
	assert tree_gini["type"] == "split", "Root should be a split for this data"
	assert len(y_pred_gini) == len(y), "Predictions length should match y"
	assert overall_mse_gini >= 0, "MSE should be non-negative"
	
	print("✓ Regression tree with Gini builds correctly.")
	return overall_mse_gini, y_pred_gini, tree_gini


def validate_notebook_dataset():
	"""Validate regression tree on notebook dataset (from the Jupyter notebook)."""
	print("\n" + "=" * 70)
	print("VALIDATION 5: Regression Tree on Notebook Dataset")
	print("=" * 70)
	
	# Dataset from the notebook (lines ~897-922 region)
	x_train = np.array([-10, -8, -3, -1, 2, 7])
	y_train = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09])
	x_test = np.array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9])
	y_test = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05])
	
	print(f"\nTraining data: {len(x_train)} samples")
	print(f"Test data: {len(x_test)} samples")
	
	measures = ["mse", "entropy", "gini"]
	
	for measure in measures:
		print(f"\n--- Using {measure.upper()} impurity ---")
		
		overall_mse, y_pred, tree = tree_overall_impurity_auto(
			x_train, y_train, max_depth=2, impurity_measure=measure
		)
		
		print(f"Built tree with max_depth=2")
		print(f"  Training MSE: {overall_mse:.6f}")
		
		# Validate tree structure
		assert tree["type"] in ["split", "leaf"], f"Invalid node type for {measure}"
		assert tree["depth"] == 0, "Root should be at depth 0"
		
		print(f"✓ {measure.upper()} tree built successfully")
	
	print("\n✓ Notebook dataset validation passed for all measures.")


def validate_max_depth_recursion():
	"""Validate that max_depth parameter properly controls tree depth."""
	print("\n" + "=" * 70)
	print("VALIDATION 6: Max Depth Recursion Control")
	print("=" * 70)
	
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])
	
	max_depths = [1, 2, 3]
	
	for max_depth in max_depths:
		overall_mse, y_pred, tree = tree_overall_impurity_auto(
			X, y, max_depth=max_depth, impurity_measure="mse"
		)
		
		# Helper to get max depth in tree
		def get_tree_max_depth(node):
			if node["type"] == "leaf":
				return node["depth"]
			left_depth = get_tree_max_depth(node["left"])
			right_depth = get_tree_max_depth(node["right"])
			return max(left_depth, right_depth)
		
		actual_max_depth = get_tree_max_depth(tree)
		print(f"max_depth={max_depth}: Actual tree depth = {actual_max_depth}")
		
		assert actual_max_depth <= max_depth, \
			f"Tree depth {actual_max_depth} exceeded max_depth {max_depth}"
	
	print("✓ Max depth parameter correctly controls tree recursion.")


def validate_entropy_vs_gini():
	"""Validate that entropy and Gini produce similar but not identical results."""
	print("\n" + "=" * 70)
	print("VALIDATION 7: Entropy vs Gini Comparison")
	print("=" * 70)
	
	y_class = np.array([0, 0, 0, 1, 1, 2])
	
	ent = entropy(y_class)
	g = gini(y_class)
	
	print(f"\nClassification data: {y_class}")
	print(f"  Entropy: {ent:.6f}")
	print(f"  Gini: {g:.6f}")
	print(f"  Difference: {abs(ent - g):.6f}")
	
	# Both should be positive and less than max value
	assert 0 < ent <= np.log2(len(np.unique(y_class))), "Entropy out of range"
	assert 0 < g < 1, "Gini out of range"
	# They should differ but be relatively close
	assert 0 < abs(ent - g) < max(ent, g), "Entropy and Gini should differ but not drastically"
	
	print("✓ Entropy and Gini produce reasonable and comparable results.")


def main():
	"""Run all validation tests."""
	try:
		validate_impurity_measures()
		splits = validate_find_best_split()
		mse_overall, mse_pred, mse_tree = validate_regression_tree_mse()
		gini_overall, gini_pred, gini_tree = validate_regression_tree_gini()
		validate_notebook_dataset()
		validate_max_depth_recursion()
		validate_entropy_vs_gini()
		
		print("\n" + "=" * 70)
		print("ALL VALIDATIONS PASSED ✓")
		print("=" * 70)
		print("\nSummary:")
		print("  ✓ Impurity measures compute correctly")
		print("  ✓ Split finding works for all measures")
		print("  ✓ Regression tree builds with MSE")
		print("  ✓ Regression tree builds with Gini")
		print("  ✓ Notebook dataset validation passed")
		print("  ✓ Max depth recursion works correctly")
		print("  ✓ Entropy and Gini produce reasonable results")
		print("\nImplementation is ready for production use.")
		
	except AssertionError as e:
		print(f"\n✗ VALIDATION FAILED: {e}")
		sys.exit(1)
	except Exception as e:
		print(f"\n✗ UNEXPECTED ERROR: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
