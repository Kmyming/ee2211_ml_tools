# EE2211 Machine Learning and Optimisation Algorithims

*Original repository forked from [Annie-spaces](https://github.com/Annie-spaces/EE2211-Machine-Learning-Tool) - Thank you for the incredible foundation!*

The most precious thing in EE2211 exam is time.

This cheatsheet provides a centralized interface for easily applying and verifying the computational methods and algorithms taught in EE2211. All examples and implementations are centralized, verified, and accessible through one file: `main.py`.

## Features and Capabilities
Here is a list of everything this repository can do, mapped to their computational objectives and source files:

**ML Regression Models:**
- Linear regression -> `LinearRegression.py`
- Polynomial regression -> `PolynomialRegression.py`
- One-hot encoding linear regression -> `OneHotLinearClassification.py`
- One-hot encoding polynomial regression -> `OneHotPolynomialRegression.py`
- Ridge regression -> `RidgeRegression.py`
- Ridge polynomial regression -> `RidgePolynomialRegression.py`
- Ridge one-hot encoding linear regression -> `RidgeOneHot.py`
- Ridge one-hot polynomial regression -> `RidgePolyOneHot.py`

**Optimisation Techniques & Algorithms:**
- Pearson's r coefficient from data -> `pearson_correlation.py`
- Bias-variance trade-off tracking (generate MSE loops across polynomial orders with and without regularisation) -> `bias_variance_tradeoff.py`
- Activation functions (sigmoid and ReLU) -> `activation_functions.py`
- Gradient descent for 1 variable (can edit iterations and learning rate) -> `gradient_descent_one_variable.py`
- Gradient descent for 2 variables (can edit iterations and learning rate) -> `gradient_descent_two_variable.py`
- Calculate node and depth impurity using the 3 methods (Gini, Entropy, Misclassification error) -> `decision_tree_impurity.py`
- Impurity measure calculations (MSE, Gini, Entropy, Misclassification Rate) -> `impurity_measures.py`
- Calculate impurity for decision tree with original data and thresholds (supports multiple impurity measures - MSE, Gini, Entropy, Misclassification Rate) -> `decision_tree.py`
- Automatic regression tree with best-split search using any impurity measure (MSE, Gini, Entropy, Misclassification Rate) and max-depth control -> `regression_tree.py`

**Unsupervised Learning:**
- K-means clustering to predict centroids (can edit iterations) -> `k_means_clustering.py`

**Matrix & Statistical Calculations:**
- Matrix invertibility, rank, determinants, linear independence, and linear equation solvers -> `matrix_functions.py`
- Covariance and standard deviation -> `statistical_functions.py`

## General Usecase
To run a specific calculation or algorithmic example:
1. Open [`main.py`](main.py).
2. At the top of the file, modify the `SELECTED_EXAMPLE` variable to match the example you want to test.
3. Locate the corresponding `run_<example>_example()` function in the same file.
4. Modify the static variables inside that example function (e.g., `X`, `Y`, `feature`, `target`) to test your own data.
5. Run `python main.py` in your terminal.

---

## 1. Matrix Examples
All inputs correspond to matrix structures typically formatted using `numpy.array()` or lists of lists.

* **`linear_independence`**: Defines `vectors = [[...], [...]]` to check if a system of row vectors are linearly independent.
* **`rank`**: Modifies the 2D matrix `X`. Outputs the calculated matrix rank value.
* **`determinant`**: Input a square matrix `X`. Returns the determinant calculation.
* **`inverse`**: Input an invertible square matrix `X`. Returns the inverted equivalent.
* **`invertibility`**: Provide `X`. Tests and confirms left/right invertibility metrics of the matrix.
* **`rank_condition`**: Provide matrix `X` and target `y`. Verifies rank condition bounds (RC checker).
* **`solve_le`**: Provide `X` and `y`. Solves systems of linear algebraic equations using the model $Xw = y$.

## 2. Statistical Examples
Inputs deal with flat 1D lists of identical length, primarily denoted as `feature` and `target`.

* **`covariance`**: Change lists `feature` and `target` to return the population-scale covariance matrix intersection.
* **`standard_deviation`**: Modify the `feature` list to query standard deviation under population basis.
* **`pearson_r`**: Input strictly parallel `feature` (the $x$ variable) and `target` (the $y$ variable) arrays to map Pearson's correlation coefficient $r$.

## 3. Regression Examples
All regression modules leverage standard training inputs `X` (raw shapes), targets `Y`, and an eventual tester bounds dataset `X_test`. In functions explicitly requiring `X_fitted` or `X_test_fitted`, the `main.py` functions use `np.hstack` to append a column of ones so the models can train an unscaled bias (intercept) parameter correctly.

* **`linear_regression`**: Requires `X_fitted` (feature matrix with appended 1s for bias), `Y` (target vector), and `X_test_fitted` (test cases testing the resultant regression line).
* **`polynomial_regression`**: Requires raw un-fitted features `X` and targets `Y`, your desired polynomial degree `order` (e.g., $2$), and `X_test`. The function handles generating the specific polynomial transforms internally.
* **`ridge_regression`**: Requires `X_fitted`, targets `Y`, a custom regularisation penalty float `LAMBDA` (e.g., $0.1$), and `X_test_fitted`. Useful for minimizing coefficient explosion in overdetermined systems.
* **`ridge_polynomial_regression`**: Uses raw un-fitted features `X` and targets `Y`, regularisation penalty float `LAMBDA`, your target polynomial degree `order`, the calculation mode `form`, and `X_test`. 
* **`onehot_linear_classification`**: Takes `X_fitted` and mappings `Y` corresponding exactly to grouped one-hot formats (e.g. $[1,0]$, $[0,1]$ mapping distinct target classes), alongside `X_test_fitted`. Outputs optimal class assignments by isolating the argument maximums of mapping outputs.
* **`onehot_polynomial_regression`**: Needs raw `X`, one-hot classification mappings `Y`, a specific degree `order`, and `X_test`. Functions similarly to its linear counterpart but internally routes points into transformed polynomial dimensions.
* **`ridge_onehot_linear_classification`**: Integrates fitted matrices `X_fitted`, one-hot target bounds `Y`, an $L_2$ regularisation float scale `LAMBDA`, polynomial base argument `order` (retained for pipeline symmetry), mode specifier `form`, and `X_test_fitted`.
* **`ridge_onehot_polynomial_regression`**: Requires raw un-fitted features `X`, one-hot targets `Y`, regularisation shrink variable `LAMBDA`, polynomial degree `order`, calculation selector `form`, and raw cases `X_test`.

**Note on the `form` input:**
For any function leveraging Ridge calculations with parameter `form` (`ridge_polynomial_regression`, `ridge_onehot_linear_classification`, `ridge_onehot_polynomial_regression`), this string tells the system exactly which algorithmic path to execute:
* `"auto"`: A detector mode estimating dimensionality. The code determines algorithmically if the system's mapping is underdetermined or overdetermined.
* `"primal"`: Enforces resolving Ridge computation using the *Primal Form* $w = (X^TX + \lambda I)^{-1}X^Ty$. Preferred when resolving computationally large samples $N$ with relatively fewer total dimensions $D$ (Overdetermined Systems).
* `"dual"`: Enforces resolving Ridge computation using the *Dual Form* $w = X^T(XX^T + \lambda I)^{-1}y$. Used optimally when you are scaling very few samples spanning an excessive number of computed dimensions (Underdetermined Systems).

## 4. Optimisation Examples
Functions leverage mathematical stepping bounds. Requires defining a target mathematical object function (`func`) and its positional derivative (`grad_func`).

* **`activation_functions`**: Applies the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ and the ReLU function $\max(0, x)$ elementwise to an input array. The example uses a small sample array and prints both outputs.

* **`bias_variance_tradeoff`**: Uses training sequences `x` and `y`, test sequences `xt` and `yt`, a maximum polynomial order `max_order`, and optional regularisation `reg`. It prints the training/test MSE for polynomial orders from 1 to `max_order`, and prints the regularised training loss when `reg` is enabled.
* **`gradient_descent_1d`**: Uses a mathematical function $f(x)$ and $f'(x)$. Adjust `start_value`, `learning_rate`, and `iterations` directly via the function call.
* **`gradient_descent_2d`**: Uses $f(x, y)$ alongside partial derivatives. Accepts custom `start_x`, `start_y`, learning rate steps, and sequence bounds constraints.

## 5. Tree Examples
Decision tree and regression tree analysis with support for multiple impurity measures.

### Impurity Measures
All tree functions support multiple impurity measures for split selection:
- **`"mse"`** (default): Mean Squared Error, ideal for regression trees with continuous targets.
- **`"entropy"`**: Shannon entropy, typically for classification trees with discrete targets. Formula: $-\sum p_i \log_2(p_i)$
- **`"gini"`**: Gini impurity, also for classification. Formula: $1 - \sum p_i^2$ (often faster than entropy).
- **`"misclassification"`**: Misclassification rate, simplest classification measure. Formula: $1 - \max(p_i)$

### Tree Examples

* **`impurity`**: Provide class counts per layer array (e.g., `layers = [[[8, 5, 5]], ...]`). Calculates the structural Gini, entropy, and misclassification error levels.

* **`decision_tree`**: Uses manually-defined thresholds to split the tree. Default example shows **MSE-based splitting**. A commented alternative demonstrates **Gini-based splitting** for comparison. You can uncomment the Gini example to see how different impurity measures affect tree structure with the same dataset.

* **`regression_tree`**: Uses one feature array `X`, target values `y`, and a `max_depth`. Automatically searches for the best split threshold at each node. Default example uses **MSE criterion**. A commented alternative shows **Gini-based automatic splits** to demonstrate how the splitting strategy changes with different impurity measures.

## 6. Clustering Examples
Unsupervised prediction techniques.

* **`kmeans`**: Supply clustering dataset variable `X` and raw starting centroid definitions via `init_centroids`. Adjust the `max_iter` parameter in the bounds call to verify step convergence boundaries along loops.

