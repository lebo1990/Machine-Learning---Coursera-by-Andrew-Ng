# from numpy import loadtxt, zeros, ones, array, linspace, logspace, arange, sin, transpose, dot
# from pylab import scatter, show, title, xlabel, ylabel, plot, contour

from numpy import *
import numpy

from matplotlib.pyplot import *

def feature_normalize(X):
    X_norm = zeros((X.shape[0], X.shape[1]))
    mean = zeros((1, X.shape[1]))
    std = zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        mean[0, i] = numpy.mean(X[:, i])
        std[0, i] = numpy.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mean[0, i]) / std[0, i]
    return X_norm, mean, std
def compute_cost_mul(X, y, theta):
    #Number of training samples
    m = X.size

    predictions = X * theta
    sqErrors = power(predictions - y, 2)
    J = (1.0 / (2 * m)) * sum(sqErrors)

    return J
def gradient_descent_mul(X, y, theta, alpha, iterations):
    m = X.shape[0]
    J_history = matrix(zeros(shape=(iterations, 1)))
    for iter in range(iterations):
        predictions = X * theta
        theta_tmp = matrix(zeros(shape = (X.shape[1], 1)))
        for i in range(X.shape[1]):
            theta_tmp[i, 0] = theta[i, 0] - 1.0 * alpha / m * sum(multiply((predictions - y), X[:, i]))
        theta = theta_tmp

        J_history[iter, 0] = compute_cost_mul(X, y, theta)

    return theta, J_history

#Load the dataset
data = (loadtxt('ex1data2.txt', delimiter=','))
y = matrix(data[:, data.shape[1] - 1]).transpose()
m = data.shape[0]
X0 = data[:,[row for row in range(data.shape[1] - 1)]]

# X with x0 = 1
X = matrix(numpy.append(ones(shape = (m, 1)), X0, 1))

# Normalize
X_norm0, mean0, std0 = feature_normalize(X0)
X_norm = matrix(numpy.append(ones(shape = (m, 1)), X_norm0, 1))


#Initialize theta parameters
mean = matrix(zeros((1, data.shape[1])))
std = matrix(zeros((1, data.shape[1])))

#Some gradient descent settings
iterations = 1000
alpha = 0.01

theta = matrix(zeros(shape=(data.shape[1], 1)))
theta, J_history = gradient_descent_mul(X_norm, y, theta, alpha, iterations)

#Plot the data
figure()
plot(J_history)
show()

#Estimate the price of a 1650 sq-ft, 3 br house using gradient descent
mean[0, 0] = 0
std[0, 0] = 1
for i in range(1, data.shape[1]):
    mean[0, i] = mean0[0, i - 1]
    std[0, i] = std0[0, i - 1]

price = divide((matrix([1, 1650, 3]) - mean), std) * theta
print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent is', price[0, 0])

#Estimate the price of a 1650 sq-ft, 3 br house using normal equations
theta_norm = linalg.pinv((X.transpose() * X)) * X.transpose() * y    # theta = (X.transpose * X)^-1 * X.transpose * y
price_norm = matrix([1, 1650, 3]) * theta_norm
print('Predicted price of a 1650 sq-ft, 3 br house using normal equations is', price_norm[0, 0])