import numpy as np
from sklearn.metrics import mean_squared_error
from impurity_measures import calculate_impurity, _as_1d_array


def build_tree(thresholds):
	tree = {}
	# Step-by-step:
	# 1. Start with an empty dict representing the root.
	# 2. For each (path, threshold) tuple, walk/create nested dicts for 'L'/'R' characters.
	# 3. At the final node set the 'TH' key to store the threshold value.
	for path, threshold in thresholds:
		node = tree
		for ch in path.upper():
			side = "left" if ch == "L" else "right"
			node = node.setdefault(side, {})
		node["TH"] = threshold
	return tree


def traverse(X, y, tree, depth, path, results, impurity_measure="mse"):
	# Step-by-step:
	# 1. Compute the impurity and mean for the current node's target values.
	# 2. If the current tree node is a leaf (no 'TH'), record leaf stats and return.
	# 3. Otherwise, split data by threshold into left/right subsets.
	# 4. If either side is empty, treat current node as a leaf (warning) and return.
	# 5. Compute impurity for left and right, compute weighted overall impurity for this split.
	# 6. Append a 'split' record to results and recurse on left and right children.
	node_impurity = calculate_impurity(y, impurity_measure)
	node_mean = np.mean(y)

	if depth not in results:
		results[depth] = []

	if not tree or "TH" not in tree:
		results[depth].append(
			{
				"path": path or "root",
				"type": "leaf",
				"n": len(y),
				"mean": node_mean,
				"impurity": node_impurity,
			}
		)
		return

	threshold = tree["TH"]
	left_mask = X <= threshold
	right_mask = X > threshold
	y_left, y_right = y[left_mask], y[right_mask]
	X_left, X_right = X[left_mask], X[right_mask]

	if len(y_left) == 0 or len(y_right) == 0:
		print(f"WARNING at '{path}': TH={threshold} puts all data on one side.")
		results[depth].append(
			{
				"path": path or "root",
				"type": "leaf",
				"n": len(y),
				"mean": node_mean,
				"impurity": node_impurity,
			}
		)
		return

	impurity_left = calculate_impurity(y_left, impurity_measure)
	impurity_right = calculate_impurity(y_right, impurity_measure)
	overall = len(y_left) / len(y) * impurity_left + len(y_right) / len(y) * impurity_right

	results[depth].append(
		{
			"path": path or "root",
			"type": "split",
			"TH": threshold,
			"n": len(y),
			"mean": node_mean,
			"impurity": node_impurity,
			"nL": len(y_left),
			"meanL": np.mean(y_left),
			"impurityL": impurity_left,
			"nR": len(y_right),
			"meanR": np.mean(y_right),
			"impurityR": impurity_right,
			"overall_impurity": overall,
		}
	)

	traverse(X_left, y_left, tree.get("left", {}), depth + 1, (path or "root") + "->L", results, impurity_measure)
	traverse(X_right, y_right, tree.get("right", {}), depth + 1, (path or "root") + "->R", results, impurity_measure)


def tree_impurity_summary(X, y, splits_flat, max_depth=0, impurity_measure="mse"):
	"""
	Print depth-by-depth breakdown of tree splits and impurity.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		splits_flat: List of (path, threshold) tuples defining the tree structure.
		max_depth: Maximum depth to report (0 means report all depths).
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		dict: Nested structure of results by depth.
	"""
	# Step-by-step:
	# 1. Compute root impurity for the full dataset.
	# 2. Build a nested tree dict from the flat splits list.
	# 3. Traverse the tree to collect per-depth node summaries.
	# 4. For each depth, compute weighted depth impurity and print detailed node info.
	root_impurity = calculate_impurity(y, impurity_measure)
	tree = build_tree(splits_flat)
	results = {}
	traverse(X, y, tree, 0, "", results, impurity_measure)

	depths = [d for d in sorted(results.keys()) if 1 <= d <= max_depth] if max_depth > 0 else sorted(results.keys())

	for depth in depths:
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

	return results


def tree_overall_impurity(X, y, splits_flat, impurity_measure="mse"):
	"""
	Calculate overall tree impurity and leaf predictions.
	
	Args:
		X: 1D array of feature values.
		y: 1D array of target values.
		splits_flat: List of (path, threshold) tuples defining the tree structure.
		impurity_measure: One of "mse", "entropy", "gini", "misclassification".
	
	Returns:
		tuple: (overall_impurity, y_pred)
	"""
	# Step-by-step:
	# 1. Build the nested tree and assign leaf predictions for each sample.
	# 2. Compute overall MSE + MSE reduction.
	# 3. Compute overall selected impurity (weighted leaf impurity) + selected reduction.
	tree = build_tree(splits_flat)
	X_arr = np.asarray(X, dtype=float).ravel()
	y_arr = np.asarray(y, dtype=float).ravel()
	y_pred = np.zeros(len(y_arr))

	def assign_predictions(X_subset, y_subset, tree_subset, idx):
		# Step-by-step for assigning predictions:
		# 1. If current subtree has no threshold, it's a leaf: set prediction to the mean of y_subset.
		# 2. Otherwise, split the subset by threshold and recursively assign to children using the same index mask.
		if not tree_subset or "TH" not in tree_subset:
			y_pred[idx] = np.mean(y_subset)
			return

		threshold = tree_subset["TH"]
		left_mask = X_subset <= threshold
		right_mask = X_subset > threshold

		if left_mask.sum() > 0:
			assign_predictions(X_subset[left_mask], y_subset[left_mask], tree_subset.get("left", {}), idx[left_mask])
		if right_mask.sum() > 0:
			assign_predictions(X_subset[right_mask], y_subset[right_mask], tree_subset.get("right", {}), idx[right_mask])

	def weighted_leaf_impurity(X_subset, y_subset, tree_subset):
		# Compute weighted selected impurity across final leaves.
		if len(y_subset) == 0:
			return 0.0

		if not tree_subset or "TH" not in tree_subset:
			return (len(y_subset) / len(y_arr)) * calculate_impurity(y_subset, impurity_measure)

		threshold = tree_subset["TH"]
		left_mask = X_subset <= threshold
		right_mask = X_subset > threshold

		if not left_mask.any() or not right_mask.any():
			return (len(y_subset) / len(y_arr)) * calculate_impurity(y_subset, impurity_measure)

		left_value = weighted_leaf_impurity(X_subset[left_mask], y_subset[left_mask], tree_subset.get("left", {}))
		right_value = weighted_leaf_impurity(X_subset[right_mask], y_subset[right_mask], tree_subset.get("right", {}))
		return left_value + right_value

	assign_predictions(X_arr, y_arr, tree, np.arange(len(y_arr)))
	
	root_mse = calculate_impurity(y_arr, "mse")
	overall_mse = mean_squared_error(y_arr, y_pred)
	mse_reduction = root_mse - overall_mse

	root_selected_impurity = calculate_impurity(y_arr, impurity_measure)
	overall_selected_impurity = weighted_leaf_impurity(X_arr, y_arr, tree)
	selected_reduction = root_selected_impurity - overall_selected_impurity

	print(f"  ROOT MSE (no split): {round(root_mse, 4)}")
	print(f"  Leaf predictions: {np.round(y_pred, 4)}")
	print(f"  Actual y        : {np.round(y_arr, 4)}")
	print(f"  Overall MSE     : {round(overall_mse, 4)}")
	print(f"  MSE reduction   : {round(mse_reduction, 4)}")
	print(f"  ROOT {impurity_measure.upper()} (no split): {round(root_selected_impurity, 4)}")
	print(f"  Overall {impurity_measure.upper()} impurity: {round(overall_selected_impurity, 4)}")
	print(f"  {impurity_measure.upper()} reduction      : {round(selected_reduction, 4)}")

	return overall_mse, y_pred


if __name__ == "__main__":
	X = np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9])
	y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5])
	TH = [("", 5)]

	print(f"{'=' * 60}")
	print(f"OVERALL TREE MSE (MSE-based splitting)")
	print(f"{'=' * 60}")
	tree_overall_impurity(X, y, TH, impurity_measure="mse")

	print(f"\n{'=' * 60}")
	print(f"DEPTH BREAKDOWN (MSE-based splitting)")
	print(f"{'=' * 60}")
	tree_impurity_summary(X, y, TH, max_depth=1, impurity_measure="mse")
