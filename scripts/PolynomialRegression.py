# FOR POLYNOMIAL REGRESSION, REMOVE BIAS COLUMN
# reshaped = X[:,1].reshape(len(X[:,1]),1)
# reshaped = Xt[:,1].reshape(len(Xt[:,1]),1)

def calculate_polynomial_parameters(num_features, polynomial_degree):
	"""Calculate the total number of parameters (columns) in polynomial regression design matrix.

	Step-by-step:
	1. Use the combinatorial formula C(d+n, n) = (d+n)! / (n! * d!)
	   where d = number of original input features and n = polynomial degree.
	2. This includes the bias term (the constant 1 in the first column).
	
	Args:
		num_features: int, number of original input features (excluding bias).
		polynomial_degree: int, polynomial degree n.
	
	Returns:
		int: Total number of parameters/columns including bias.
	"""
	from math import comb
	# C(d+n, n) computes binomial coefficient
	total_params = comb(num_features + polynomial_degree, polynomial_degree)
	return total_params


def polynomial_regression(X,y,order,X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    # Step-by-step:
    # 1. Transform raw features X into polynomial features P using `PolynomialFeatures`.
    # 2. Solve for polynomial coefficients w using closed-form linear algebra depending on rank.
    # 3. Compute training predictions and errors, transform X_test to P_test and produce test predictions.
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X) # learns which new feature combinations to create based on the input dimensions) and then applies this transformation to the same data.
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if P.shape[1] < P.shape[0]:
        system = "overdetermined"
    elif P.shape[1] > P.shape[0]:
        system = "underdetermined"
    else:
        system = "full rank"
    print(system, "system")
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if system == "overdetermined":
        w = np.linalg.inv(P.T @ P) @ P.T @ y
    elif system == "underdetermined":
        w = P.T @ np.linalg.inv(P @ P.T) @ y
    else:
        w = np.linalg.inv(P) @ y
    print("w is: ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is: ", P_train_predicted)
    print("y_train_classified is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    P_test = poly.transform(X_test) # applies the same transformation to the test data, ensuring that the same polynomial features are created for the test set as were created for the training set.
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_predicted is")
    print(y_predicted)
    print("y_predicted_classified is\n", np.sign(y_predicted), "\n")


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #
    # return(system, P, w, y_predicted, y_classified)

    # print("P rank:", np.linalg.matrix_rank(P))
    # result=np.hstack((P,y))
    # print("P|y rank: ", np.linalg.matrix_rank(result))

if __name__ == "__main__":
    # sample data
    import numpy as np

    X = np.array([[1,1],[2,1],[1,2],[2,3]])
    Y = np.array([[2],[3.1],[3.5],[4]])
    X_fitted=np.hstack((np.ones((len(X),1)),X))
    X_test = np.array([[1,-2]])
    X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
    polynomial_regression(X,Y,2,X_test)