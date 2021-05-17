def gradientDescentReg(X,y,theta,lamb, num_iters, alpha):
    m = len(y)
    J_hist = np.zeros((num_iters,1))
    J_theta = np.zeros((num_iters,3))
    for iter in range(0, num_iters):
        h = sigmoid(np.dot(X,theta))
        grad = np.dot(np.transpose(X),(h-y))/m
        grad_reg = lamb*theta
        grad_reg[0] = 0
        grad = grad + grad_reg
        theta = theta - alpha*grad*(1/m)
        J_hist[iter] = costFuncReg(theta, X, y, lamb)
        J_theta[iter][0] = theta[0]
        J_theta[iter][1] = theta[1]
        J_theta[iter][2] = theta[2]
    for i in range (0, num_iters):
        if (np.min(J_hist) == J_hist[i]):
            return (J_theta[i][:])