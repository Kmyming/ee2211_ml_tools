def ridge_linear_regression_onehot(X,y,LAMBDA,order, form, X_test):
    import numpy as np
    if form=="auto":
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")

    if form=="primal form":
        I = np.identity(X.shape[1])
        w = np.linalg.inv(X.T @ X+LAMBDA*I) @ X.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = X.T @ np.linalg.inv(X @ X.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(X) @ y

    print("w is: ")
    print(w)
    print("")

    y_calculated=X@w
    print("y calculated is: \n", y_calculated, "\n")
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    print("y_predicted is\n", y_predicted, "\n")
    y_predicted=np.argmax(y_predicted,axis=1) # Maximum index along axis 1 (rows), outputs the index the max value is located at
    print("y_predicted class is\n", y_predicted, "\n")

if __name__ == "__main__":
    # sample data
    import numpy as np

    X = np.array([[2,1,0],[0,3,1],[1,0,3],[3,1,4],[-1,2,1]])
    Y = np.array([[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0]])
    X_fitted=np.hstack((np.ones((len(X),1)),X))
    X_test = np.array([[1,1,2]])
    X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
    ridge_linear_regression_onehot(X_fitted, Y, 0.01, 2, "auto", X_test_fitted)