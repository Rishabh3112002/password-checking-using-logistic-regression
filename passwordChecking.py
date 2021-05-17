import numpy as np
X = np.array([[1,8,1],[1,12,2],[1,1,1],[1,1,0],[1,8,0],[1,12,4],[1,7,1]])
y = np.array([[1],[1],[0],[0],[0],[1],[1]])
theta = np.array([[0],[0],[0]])
lamb = 1.5
alpha = 0.1
num_iters = 5000
def sigmoid(z):
    g = np.zeros(len(z))
    for i in range(0, len(z)):
        g = np.ones(len(z))*np.transpose(1/(1.0+np.exp(-z)))
        g = np.transpose(g)
    return g

def costFuncReg(theta, X, y, lamb):
    h = np.zeros(len(X))
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X,theta))
    h = np.array(h)
    cost = np.sum(-y* np.log(h) - ((1-y)*np.log(1-h)))
    J = cost / m + (lamb / (2.0 * m)) * np.square(np.sum(theta[1:len(theta)]))
    return J

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

a = gradientDescentReg(X,y,theta,lamb, num_iters, alpha)
costFuncReg(theta, X, y, lamb)
sigmoid(np.dot(X,theta))
passwrd = input('Enter a password')
le = len(passwrd)
c = int(0)
for i in range (0,le):
    if ord(passwrd[i])>=65 and ord(passwrd[i])<=90:
        c = c+1
ch = -a[0]
sh = a[1]*le + a[2]*c
print(sh)
print(ch)
if sh>=ch:
    print('Strong Password')
else:
    print('Weak password')
