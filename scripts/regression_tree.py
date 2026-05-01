import numpy as np
from sklearn.metrics import mean_squared_error
from impurity_measures import calculate_impurity, _as_1d_array


def find_best_split(X, y, impurity_measure="mse"):
	"""
	Find the best split threshold that minimizes impurity.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		dict: Split information including threshold, impurities, and data subsets.
		None: If no valid split exists.
	"""
	# Step-by-step:
	# 1. Ensure X and y are 1D numpy arrays of equal length.
	# 2. Compute candidate thresholds as midpoints between unique, sorted X values.
	# 3. For each candidate threshold: split y into left/right, compute impurity for each side,
	#    and compute weighted impurity. Track the threshold minimizing weighted impurity.
	# 4. Return a dict describing the best split (threshold, weighted impurity, subsets) or None.
	X = _as_1d_array(X, "X")
	y = _as_1d_array(y, "y")
	if X.size != y.size:
		raise ValueError("X and y must have the same number of samples.")

	unique_x = np.unique(X)
	if unique_x.size < 2:
		return None

	thresholds = (unique_x[:-1] + unique_x[1:]) / 2.0
	best = None
	current_impurity = calculate_impurity(y, impurity_measure)

	for threshold in thresholds:
		left_mask = X <= threshold
		right_mask = X > threshold
		if not left_mask.any() or not right_mask.any():
			continue

		y_left, y_right = y[left_mask], y[right_mask]
		impurity_left = calculate_impurity(y_left, impurity_measure)
		impurity_right = calculate_impurity(y_right, impurity_measure)
		weighted_impurity = (len(y_left) / len(y)) * impurity_left + (len(y_right) / len(y)) * impurity_right

		if best is None or weighted_impurity < best["weighted_impurity"]:
			best = {
				"threshold": float(threshold),
				"weighted_impurity": float(weighted_impurity),
				"left_X": X[left_mask],
				"left_y": y_left,
				"right_X": X[right_mask],
				"right_y": y_right,
				"left_impurity": float(impurity_left),
				"right_impurity": float(impurity_right),
			}

	if best is None:
		return None

	best["node_impurity"] = float(current_impurity)
	return best


def build_regression_tree(X, y, depth=0, max_depth=1, path="root", impurity_measure="mse"):
	"""
	Recursively build a regression tree with automatic threshold selection.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		depth: Current depth (used internally for recursion).
		max_depth: Maximum tree depth.
		path: Current node path (used internally for tracking).
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		dict: Tree node structure.
	"""
	# Step-by-step:
	# 1. Normalize inputs to 1D arrays and compute node-level statistics (mean, impurity, count).
	# 2. If stopping criteria met (max depth, single sample, constant targets) -> make a leaf with prediction=mean.
	# 3. Otherwise, call `find_best_split` to get the best threshold.
	# 4. If no beneficial split found (or split doesn't reduce impurity), return a leaf.
	# 5. If a split is found, mark node as 'split', store threshold and overall impurity, and recurse to build left/right children.
	X = _as_1d_array(X, "X")
	y = _as_1d_array(y, "y")
	if X.size != y.size:
		raise ValueError("X and y must have the same number of samples.")

	node_mean = float(np.mean(y)) if y.size else 0.0
	node_impurity = float(calculate_impurity(y, impurity_measure))
	node = {
		"path": path,
		"depth": depth,
		"n": int(len(y)),
		"mean": node_mean,
		"impurity": node_impurity,
	}

	if depth >= max_depth or y.size <= 1 or np.allclose(y, y[0]):
		node["type"] = "leaf"
		node["prediction"] = node_mean
		return node

	best = find_best_split(X, y, impurity_measure)
	if best is None or best["weighted_impurity"] >= node_impurity - 1e-12:
		node["type"] = "leaf"
		node["prediction"] = node_mean
		return node

	node["type"] = "split"
	node["threshold"] = best["threshold"]
	node["overall_impurity"] = best["weighted_impurity"]
	node["left"] = build_regression_tree(best["left_X"], best["left_y"], depth + 1, max_depth, path + "->L", impurity_measure)
	node["right"] = build_regression_tree(best["right_X"], best["right_y"], depth + 1, max_depth, path + "->R", impurity_measure)
	return node


def _predict_one(x_value, node):
	"""Recursively predict for a single sample.

	Step-by-step:
	1. If the node is a leaf, return its stored prediction.
	2. Otherwise compare the sample value to the node threshold.
	3. Recurse into left child when x <= threshold, else into right child.
	"""
	if node["type"] == "leaf":
		return node["prediction"]
	if x_value <= node["threshold"]:
		return _predict_one(x_value, node["left"])
	return _predict_one(x_value, node["right"])


def predict_tree(X, tree):
	"""Generate predictions for an array of samples.

	Step-by-step:
	1. Ensure X is a 1D array.
	2. For each sample value call the single-sample recursive predictor `_predict_one`.
	3. Return a numpy array of float predictions.
	"""
	X = _as_1d_array(X, "X")
	return np.array([_predict_one(x_value, tree) for x_value in X], dtype=float)


def _collect_summary(node, results):
	"""Recursively collect node summary by depth."""
	depth = node["depth"]
	results.setdefault(depth, [])

	if node["type"] == "leaf":
		results[depth].append(
			{
				"path": node["path"],
				"type": "leaf",
				"n": node["n"],
				"mean": node["mean"],
				"impurity": node["impurity"],
			}
		)
		return

	left = node["left"]
	right = node["right"]
	results[depth].append(
		{
			"path": node["path"],
			"type": "split",
			"TH": node["threshold"],
			"n": node["n"],
			"mean": node["mean"],
			"impurity": node["impurity"],
			"nL": left["n"],
			"meanL": left["mean"],
			"impurityL": left["impurity"],
			"nR": right["n"],
			"meanR": right["mean"],
			"impurityR": right["impurity"],
			"overall_impurity": node["overall_impurity"],
		}
	)
	_collect_summary(left, results)
	_collect_summary(right, results)


def tree_overall_impurity_auto(X, y, max_depth=1, impurity_measure="mse"):
	"""
	Build tree with automatic splits and compute overall impurity.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		max_depth: Maximum tree depth.
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		tuple: (overall_mse, y_pred, tree)
	"""
	# Step-by-step:
	# 1. Normalize inputs and build the regression tree up to max_depth using automatic splits.
	# 2. Generate predictions and compute overall MSE + MSE reduction.
	# 3. Compute weighted selected impurity across leaves + selected-impurity reduction.
	X = _as_1d_array(X, "X")
	y = _as_1d_array(y, "y")
	tree = build_regression_tree(X, y, max_depth=max_depth, impurity_measure=impurity_measure)
	y_pred = predict_tree(X, tree)
	overall_mse = mean_squared_error(y, y_pred)
	root_mse = calculate_impurity(y, "mse")
	mse_reduction = root_mse - overall_mse

	def weighted_leaf_impurity(node, total_samples):
		if node["type"] == "leaf":
			return (node["n"] / total_samples) * node["impurity"]
		return weighted_leaf_impurity(node["left"], total_samples) + weighted_leaf_impurity(node["right"], total_samples)

	root_selected_impurity = tree["impurity"]
	overall_selected_impurity = weighted_leaf_impurity(tree, len(y))
	selected_reduction = root_selected_impurity - overall_selected_impurity

	print(f"  ROOT MSE (no split): {round(root_mse, 4)}")
	print(f"  Leaf predictions: {np.round(y_pred, 4)}")
	print(f"  Actual y        : {np.round(y, 4)}")
	print(f"  Overall MSE     : {round(overall_mse, 4)}")
	print(f"  MSE reduction   : {round(mse_reduction, 4)}")
	print(f"  ROOT {impurity_measure.upper()} (no split): {round(root_selected_impurity, 4)}")
	print(f"  Overall {impurity_measure.upper()} impurity: {round(overall_selected_impurity, 4)}")
	print(f"  {impurity_measure.upper()} reduction      : {round(selected_reduction, 4)}")

	return overall_mse, y_pred, tree


def tree_impurity_summary_auto(X, y, max_depth=1, impurity_measure="mse"):
	"""
	Build tree and print depth-by-depth impurity breakdown.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		max_depth: Maximum tree depth.
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		tuple: (results, tree)
	"""
	# Step-by-step:
	# 1. Normalize inputs, build the tree with automatic splits.
	# 2. Collect per-depth node summaries with `_collect_summary`.
	# 3. For each depth compute weighted impurity and print node breakdowns.
	X = _as_1d_array(X, "X")
	y = _as_1d_array(y, "y")
	tree = build_regression_tree(X, y, max_depth=max_depth, impurity_measure=impurity_measure)
	results = {}
	_collect_summary(tree, results)
	root_impurity = tree["impurity"]

	for depth in sorted(results.keys()):
		nodes = results[depth]
		depth_overall = sum(
			(node["n"] / len(y)) * (node["overall_impurity"] if node["type"] == "split" else node["impurity"])
			for node in nodes
		)
		delta = root_impurity - depth_overall
		print(f"\n--- Depth {depth} | {impurity_measure.upper()} impurity: {round(depth_overall, 4)} | Delta from root: {round(delta, 4)} ---")

		for node in nodes:
			tag = "(leaf)" if node["type"] == "leaf" else f"TH={node['TH']}"
			print(f"\n  [{node['path']}] {tag}  n={node['n']}  y_pred={round(node['mean'], 4)}  node_impurity={round(node['impurity'], 4)}")
			if node["type"] == "split":
				print(f"    left : n={node['nL']}  mean={round(node['meanL'], 4)}  impurity={round(node['impurityL'], 4)}")
				print(f"    right: n={node['nR']}  mean={round(node['meanR'], 4)}  impurity={round(node['impurityR'], 4)}")
				print(f"    overall impurity = {round(node['overall_impurity'], 4)}")

	return results, tree


if __name__ == "__main__":
	X = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
	y = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])

	print("=" * 60)
	print("AUTOMATIC REGRESSION TREE MSE (MSE-based splitting)")
	print("=" * 60)
	tree_overall_impurity_auto(X, y, max_depth=1, impurity_measure="mse")

	print(f"\n{'=' * 60}")
	print(f"DEPTH BREAKDOWN (MSE-based splitting)")
	print(f"{'=' * 60}")
	tree_impurity_summary_auto(X, y, max_depth=1, impurity_measure="mse")
