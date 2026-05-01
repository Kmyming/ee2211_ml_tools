def onehot_linearclassification(X, y, X_test):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    # Step-by-step:
    # 1. (Optional) If y is not already one-hot encoded, fit/transform using OneHotEncoder.
    # 2. Treat the one-hot columns as multiple regression targets and solve the linear systems
    #    (one weight matrix for all classes) using the same closed-form logic as linear regression.
    # 3. Compute training error metrics and predict class indices on X_test by taking argmax over class scores.
    # onehot = OneHotEncoder(sparse_output=False)
    # Y_onehot = onehot.fit_transform(y)
    # print(Y_onehot)
    # this is to be run if y is in the form of 1D array, if y is already in one-hot encoding form, then we can skip this step
    # eg y = np.array([[1],[1],[2],[3],[2]])
    # y = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1], [0,1,0]])
    # AFTER RUNNING, TAKE Y_onehot AS THE NEW y FOR THE LINEAR REGRESSION PROCESS BELOW

    #linear regression process
    if X.shape[1]<X.shape[0]:
        system="overdetermined"
    elif X.shape[1]>X.shape[0]:
        system="underdetermined"
    else:
        system="full rank"
    print(system, "system \n")

    if system=="overdetermined":
        w=np.linalg.inv(X.T@X)@X.T@y
    elif system=="underdetermined":
        w=X.T@np.linalg.inv(X@X.T)@y
    else:
        w=np.linalg.inv(X)@y
    print("w is: \n", w, "\n")

    y_calculated=X@w
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    y_predicted=np.argmax(y_predicted,axis=1) # Maximum index along axis 1 (rows), outputs the index the max value is located at
    print("y_predicted is\n", y_predicted, "\n")

if __name__ == "__main__":
    # sample data
    import numpy as np

    X = np.array([[1,1],[2,1],[1,2],[2,3]])
    Y = np.array([[2],[3.1],[3.5],[4]])
    X_fitted=np.hstack((np.ones((len(X),1)),X))
    X_test = np.array([[1,-2]])
    X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
    onehot_linearclassification(X_fitted,Y,X_test_fitted)

    # MANUAL ONE-HOT ENCODING FOR 3 CLASSES
    # Ytr_onehot = list()
    # for i in y_train:
    # letter = [0, 0, 0]
    # letter[i] = 1
    # Ytr_onehot.append(letter)
    # Yts_onehot = list()
    # for i in y_test:
    # letter = [0, 0, 0]
    # letter[i] = 1
    # Yts_onehot.append(letter)



