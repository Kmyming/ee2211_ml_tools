import numpy as np


def _node_metrics(node):
    node = np.asarray(node, dtype=float)
    probabilities = node / np.sum(node)
    safe_probabilities = probabilities[probabilities > 0]
    gini = 1 - np.sum(probabilities ** 2)
    entropy = -np.sum(safe_probabilities * np.log2(safe_probabilities))
    misclass = 1 - np.max(probabilities)
    return gini, entropy, misclass


def tree_impurity(layers):
    """Print impurity metrics for each node and weighted impurity per depth.

    Step-by-step:
    1. Iterate over depth layers supplied as `layers` (each layer is a list of class-count arrays).
    2. Convert each node to numpy arrays and compute total samples at this depth.
    3. For each node compute Gini, entropy and misclassification via `_node_metrics`.
    4. Compute the weight (node sample count / total samples) for each node.
    5. Compute depth-level (weighted) impurity by summing weight * node_impurity.
    6. Print node-level and depth-level impurity summaries and return structured summaries.
    """
    summaries = []
    for depth, sample_arr in enumerate(layers, start=0):
        if not sample_arr:
            continue

        sample_arr = [np.array(node) for node in sample_arr]
        total_samples = sum(np.sum(node) for node in sample_arr)

        node_gini_lst = []
        node_entropy_lst = []
        node_misclass_lst = []
        ratio_lst = []

        for node in sample_arr:
            gini, entropy, misclass = _node_metrics(node)
            node_gini_lst.append(gini)
            node_entropy_lst.append(entropy)
            node_misclass_lst.append(misclass)
            ratio_lst.append(np.sum(node) / total_samples)

        depth_gini = sum(r * q for r, q in zip(ratio_lst, node_gini_lst))
        depth_entropy = sum(r * q for r, q in zip(ratio_lst, node_entropy_lst))
        depth_misclass = sum(r * q for r, q in zip(ratio_lst, node_misclass_lst))

        summary = {
            "depth": depth,
            "node_gini": node_gini_lst,
            "node_entropy": node_entropy_lst,
            "node_misclass": node_misclass_lst,
            "depth_gini": depth_gini,
            "depth_entropy": depth_entropy,
            "depth_misclass": depth_misclass,
        }
        summaries.append(summary)

        print(f"\n{'=' * 40}")
        print(f"Depth {depth}" + (" (root)" if depth == 0 else ""))
        print(f"{'=' * 40}")
        print(f"Node Gini impurity:  {[round(float(x), 4) for x in node_gini_lst]}")
        print(f"Node entropy:        {[round(float(x), 4) for x in node_entropy_lst]}")
        print(f"Node misclass rate:  {[round(float(x), 4) for x in node_misclass_lst]}")
        print(f"\nDepth Gini impurity:  {round(depth_gini, 4)}")
        print(f"Depth entropy:        {round(depth_entropy, 4)}")
        print(f"Depth misclass rate:  {round(depth_misclass, 4)}")

    return summaries


def overall_impurity(layers, selected_depths):
    """Print impurity metrics across selected depths.

    Step-by-step:
    1. Collect nodes from the requested `selected_depths` into a single list.
    2. If no valid nodes found, return None.
    3. Compute total sample count across the selected nodes.
    4. For each node compute Gini, entropy and misclassification via `_node_metrics`.
    5. Compute overall (weighted) impurity using node sample proportions.
    6. Print and return a summary dictionary with node-level and overall impurity values.
    """
    all_nodes = []
    for depth in selected_depths:
        if depth >= len(layers) or not layers[depth]:
            print(f"Warning: depth {depth} is empty or out of range, skipping.")
            continue
        for node in layers[depth]:
            all_nodes.append(np.array(node, dtype=float))

    if not all_nodes:
        print("No valid nodes found for selected depths.")
        return None

    total_samples = sum(np.sum(node) for node in all_nodes)

    node_gini_lst = []
    node_entropy_lst = []
    node_misclass_lst = []
    ratio_lst = []

    for node in all_nodes:
        gini, entropy, misclass = _node_metrics(node)
        node_gini_lst.append(gini)
        node_entropy_lst.append(entropy)
        node_misclass_lst.append(misclass)
        ratio_lst.append(np.sum(node) / total_samples)

    overall_gini = sum(r * q for r, q in zip(ratio_lst, node_gini_lst))
    overall_entropy = sum(r * q for r, q in zip(ratio_lst, node_entropy_lst))
    overall_misclass = sum(r * q for r, q in zip(ratio_lst, node_misclass_lst))

    summary = {
        "node_gini": node_gini_lst,
        "node_entropy": node_entropy_lst,
        "node_misclass": node_misclass_lst,
        "overall_gini": overall_gini,
        "overall_entropy": overall_entropy,
        "overall_misclass": overall_misclass,
    }

    print(f"\n{'=' * 40}")
    print(f"Overall Impurity (depths {selected_depths})")
    print(f"{'=' * 40}")
    print(f"Node Gini impurity:  {[round(float(x), 4) for x in node_gini_lst]}")
    print(f"Node entropy:        {[round(float(x), 4) for x in node_entropy_lst]}")
    print(f"Node misclass rate:  {[round(float(x), 4) for x in node_misclass_lst]}")
    print(f"\nOverall Gini impurity:  {round(overall_gini, 4)}")
    print(f"Overall entropy:        {round(overall_entropy, 4)}")
    print(f"Overall misclass rate:  {round(overall_misclass, 4)}")

    return summary


if __name__ == "__main__":
    layers = [
        [[8, 5, 5]],
        [[6, 4, 0], [2, 1, 5]],
        [],
    ]

    tree_impurity(layers)
    overall_impurity(layers, selected_depths=[1])
