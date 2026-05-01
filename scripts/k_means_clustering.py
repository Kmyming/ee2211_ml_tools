import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(x, init_centroids, max_iter=2, random_state=0):
    """Run KMeans once with a limited iteration count and once to convergence.

    Step-by-step:
    1. Convert inputs to numpy arrays and ensure correct dtypes.
    2. Fit a `KMeans` instance with `max_iter` to show intermediate centroids/labels.
    3. Fit another `KMeans` instance without `max_iter` to run until convergence.
    4. Return both partial and converged results (models, centroids, labels).
    """
    x = np.asarray(x, dtype=float)
    init_centroids = np.asarray(init_centroids, dtype=float)

    kmeans_partial = KMeans(
        n_clusters=len(init_centroids),
        init=init_centroids,
        n_init=1,
        max_iter=max_iter,
        random_state=random_state,
    )
    kmeans_partial.fit(x)

    kmeans_converged = KMeans(
        n_clusters=len(init_centroids),
        init=init_centroids,
        n_init=1,
        random_state=random_state,
    )
    kmeans_converged.fit(x)

    return {
        "partial": {
            "model": kmeans_partial,
            "centroids": kmeans_partial.cluster_centers_,
            "labels": kmeans_partial.labels_,
        },
        "converged": {
            "model": kmeans_converged,
            "centroids": kmeans_converged.cluster_centers_,
            "labels": kmeans_converged.labels_,
        },
    }


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]])
    init_centroids = np.array([[0, 0], [3, 0]])
    max_iter = 2
    result = run_kmeans(x, init_centroids, max_iter=max_iter)

    print(f"=== After {max_iter} Iteration(s) ===")
    print("Centroids:\n", result["partial"]["centroids"])
    print("Predictions:", result["partial"]["labels"])
    print("\n=== After Convergence ===")
    print("Centroids:\n", result["converged"]["centroids"])
    print("Predictions:", result["converged"]["labels"])
