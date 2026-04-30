import numpy as np
from sklearn.metrics import mean_squared_error


def mse(y):
    y = np.asarray(y, dtype=float).ravel()
    return mean_squared_error(y, np.ones(len(y)) * np.mean(y))


def build_tree(thresholds):
    tree = {}
    for path, threshold in thresholds:
        node = tree
        for ch in path.upper():
            side = "left" if ch == "L" else "right"
            node = node.setdefault(side, {})
        node["TH"] = threshold
    return tree


def traverse(X, y, tree, depth, path, results):
    node_mse = mse(y)
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
                "mse": node_mse,
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
                "mse": node_mse,
            }
        )
        return

    mse_left = mse(y_left)
    mse_right = mse(y_right)
    overall = len(y_left) / len(y) * mse_left + len(y_right) / len(y) * mse_right

    results[depth].append(
        {
            "path": path or "root",
            "type": "split",
            "TH": threshold,
            "n": len(y),
            "mean": node_mean,
            "mse": node_mse,
            "nL": len(y_left),
            "meanL": np.mean(y_left),
            "mseL": mse_left,
            "nR": len(y_right),
            "meanR": np.mean(y_right),
            "mseR": mse_right,
            "overall_mse": overall,
        }
    )

    traverse(X_left, y_left, tree.get("left", {}), depth + 1, (path or "root") + "->L", results)
    traverse(X_right, y_right, tree.get("right", {}), depth + 1, (path or "root") + "->R", results)


def tree_MSE_summary(X, y, splits_flat, max_depth=0):
    root_mse = mse(y)
    tree = build_tree(splits_flat)
    results = {}
    traverse(X, y, tree, 0, "", results)

    depths = [d for d in sorted(results.keys()) if 1 <= d <= max_depth]

    for depth in depths:
        nodes = results[depth]
        depth_overall = sum(
            (node["n"] / len(y)) * (node["overall_mse"] if node["type"] == "split" else node["mse"])
            for node in nodes
        )
        delta = root_mse - depth_overall
        print(f"\n--- Depth {depth} | weighted MSE: {round(depth_overall, 4)} | Delta from root: {round(delta, 4)} ---")

        for node in nodes:
            tag = "(leaf)" if node["type"] == "leaf" else f"TH={node['TH']}"
            print(f"\n  [{node['path']}] {tag}  n={node['n']}  y_pred={round(node['mean'], 4)}  node_MSE={round(node['mse'], 4)}")
            if node["type"] == "split":
                print(f"    left : n={node['nL']}  mean={round(node['meanL'], 4)}  MSE={round(node['mseL'], 4)}")
                print(f"    right: n={node['nR']}  mean={round(node['meanR'], 4)}  MSE={round(node['mseR'], 4)}")
                print(f"    overall MSE = {round(node['overall_mse'], 4)}")

    return results


def tree_overall_MSE(X, y, splits_flat):
    root_mse = mse(y)
    tree = build_tree(splits_flat)
    y_pred = np.zeros(len(y))

    def assign_predictions(X_subset, y_subset, tree_subset, idx):
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

    assign_predictions(np.asarray(X, dtype=float).ravel(), np.asarray(y, dtype=float).ravel(), tree, np.arange(len(y)))
    overall_mse = mean_squared_error(np.asarray(y, dtype=float).ravel(), y_pred)

    print(f"  ROOT MSE (no split): {round(root_mse, 4)}")
    print(f"  Leaf predictions: {np.round(y_pred, 4)}")
    print(f"  Actual y        : {np.round(np.asarray(y, dtype=float).ravel(), 4)}")
    print(f"  Overall MSE     : {round(overall_mse, 4)}")
    print(f"  MSE reduction   : {round(root_mse - overall_mse, 4)}")

    return overall_mse, y_pred


if __name__ == "__main__":
    X = np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9])
    y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5])
    TH = [("", 5)]

    print(f"{'=' * 60}")
    print(f"OVERALL TREE MSE")
    print(f"{'=' * 60}")
    tree_overall_MSE(X, y, TH)

    print(f"\n{'=' * 60}")
    print(f"DEPTH BREAKDOWN")
    print(f"{'=' * 60}")
    tree_MSE_summary(X, y, TH, max_depth=1)
