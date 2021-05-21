#Checking if a Password is strong or not using Logistic Regression Algorithum
import numpy as np
X = np.loadtxt('X.txt')#Training set
y = np.loadtxt('y.txt')
y = y.reshape(len(y),1)# Result for the training examples
theta = np.array([[0],[0],[0],[0]])
lamb = 1.5
alpha = 1
num_iters = 5000 #Value to run the number of iterations for minimizing theta
def sigmoid(z): #Computing a sigmoid function for the hypothesis
    g = np.zeros(len(z))
    for i in range(0, len(z)):
        g = np.ones(len(z))*np.transpose(1/(1.0+np.exp(-z)))
        g = np.transpose(g)
    return g

def costFuncReg(theta, X, y, lamb): # To check the trueness of theta
    h = np.zeros(len(X))
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X,theta))
    h = np.array(h)
    cost = np.sum(-y* np.log(h) - ((1-y)*np.log(1-h)))
    J = cost / m + (lamb / (2.0 * m)) * np.square(np.sum(theta[1:len(theta)])) # Computing Cost Function
    return J

def gradientDescentReg(X,y,theta,lamb, num_iters, alpha): # For finding values of theta which minimize J
    m = len(y)
    l = len(theta)
    J_hist = np.zeros((num_iters,1))
    J_theta = np.zeros((num_iters,l))
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
        J_theta[iter][3] = theta[3]
    for i in range (0, num_iters):
        if (np.min(J_hist) == J_hist[i]):
            return (J_theta[i][:])

#Taking the password as input from the user and checking if password is strong or not
a = gradientDescentReg(X,y,theta,lamb, num_iters, alpha)
costFuncReg(theta, X, y, lamb)
sigmoid(np.dot(X,theta))
passwrd = input('Enter a password')
le = len(passwrd)
c = int(0)
sp = int(0)
for i in range (0,le):
    if ord(passwrd[i])>=65 and ord(passwrd[i])<=90: # find the number of capitals in entered password
        c = c+1
    elif ord(passwrd[i])>=32 and ord(passwrd[i])<=47 or ord(passwrd[i])>=58 and ord(passwrd[i])<=64 or ord(passwrd[i])>=91 and ord(passwrd[i])<=96 or ord(passwrd[i])>=123 and ord(passwrd[i])<=126:# Find the number of special characters in entered password
        sp = sp+1
ch = -a[0]#for computing decision boundary
sh = a[1]*le + a[2]*c + a[3]*sp #for computing decision boundary
print(ch)
print(sh)
#checking password with decision boundary
if sh>=ch:
    print('Strong Password')
else:
    print('Weak password')
