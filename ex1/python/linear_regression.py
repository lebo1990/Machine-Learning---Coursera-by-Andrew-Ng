# from numpy import loadtxt, zeros, ones, array, linspace, logspace, arange, sin, transpose, dot
# from pylab import scatter, show, title, xlabel, ylabel, plot, contour

from numpy import *
from matplotlib.pyplot import *

def compute_cost(X, y, theta):
    m = y.size

    predictions = X * theta
    sqErrors = power(predictions - y, 2)
    J = (1.0 / (2 * m)) * sum(sqErrors)
    return J
def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = matrix(zeros(shape=(num_iters, 1)))

    for i in range(num_iters):
        predictions = X * theta
        theta0 = theta[0, 0] - alpha * (1.0 / m) * sum(multiply((predictions - y), X[:, 0]))
        theta1 = theta[1, 0] - alpha * (1.0 / m) * sum(multiply((predictions - y), X[:, 1]))
        theta = matrix([[theta0], [theta1]])
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

#Load the dataset
data = matrix(loadtxt('ex1data1.txt', delimiter=','))
y = matrix(data[:, 1])
m = y.size
X = matrix(ones(shape=(m, 2)))
X[:, 1] = data[:, 0]

#Initialize theta parameters
theta = matrix(zeros(shape=(2, 1)))

#Some gradient descent settings
iterations = 1000
alpha = 0.01

theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

#Plot the data
figure()
subplot(2, 1, 1)
scatter(X[:, 1], y[:, 0], marker='x', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
plot(X[:, 1], dot(X, theta), color = 'r')

subplot(2, 1, 2)
plot(J_history)
show()