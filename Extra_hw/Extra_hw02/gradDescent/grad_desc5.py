import numpy as np
import random


def read_data(infile):
    """Read the datafile and return arrays"""
    data = np.genfromtxt(infile, delimiter=None, dtype=float)
    X = data[:, :-1]
    t = data[:, [-1]]
    return X, t

# Data file paths
fh_train =  "../data/univariate/train.txt"
fh_test  = "../data/univariate/test.txt"

# Print weight vector
x,y = read_data(fh_train)
print('y.shape = ', y.shape)
x = np.append(np.ones_like(y),x, axis=1)


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print(("Iteration %d | Cost: %f" % (i, cost)))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta




# gen 100 points with a bias of 25 and 10 variance as a bit of noise
# x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations= 1000
alpha = 0.0005
theta = np.ones(n)

theta = np.array([theta]).T
print('x = ', x)
print('y = ', y)
print('theta = ', theta)
print('x.shape = ', x.shape)
print('y.shape = ', y.shape)
print('theta.shape = ', theta.shape)

# Run grad desc
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print('Final Theta = ', theta)
