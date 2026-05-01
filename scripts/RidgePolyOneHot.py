def ridge_poly_regression_onehot(X,y,LAMBDA,order, form, X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    # Step-by-step:
    # 1. Expand features into polynomial design matrix P using the requested order.
    # 2. Choose primal/dual ridge form and compute ridge weights on P (similar to ridge_regression).
    # 3. Compute training predictions and test-time predictions transforming X_test with the same polynomial mapping.
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
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

    P_test = poly.transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_predicted is")
    print(y_predicted)
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
    ridge_poly_regression_onehot(X,Y,0.01,2,"auto", X_test)
