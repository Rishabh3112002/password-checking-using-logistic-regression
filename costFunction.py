def costFuncReg(theta, X, y, lamb):
    h = np.zeros(len(X))
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X,theta))
    h = np.array(h)
    cost = np.sum(-y* np.log(h) - ((1-y)*np.log(1-h)))
    J = cost / m + (lamb / (2.0 * m)) * np.square(np.sum(theta[1:len(theta)]))
    return J