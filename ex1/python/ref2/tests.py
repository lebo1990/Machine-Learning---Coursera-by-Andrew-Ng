import unittest
from utils import hypothesis,computeCost,computeCostLoop,gradientDescent
from utils import feature_normalization_loop, feature_normalization, normal_equation

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class LinearRegressionTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = np.genfromtxt('ex1data1.txt', delimiter=',')
        self.X, self.y = self.data[:, 0], self.data[:, 1]
        self.m = len(self.y)
        self.X = self.X.reshape(self.m,1)
        self.y = self.y.reshape(self.m, 1)
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
    
    def test_cost(self):
        theta = np.zeros((2,1))
        self.assertAlmostEqual(computeCostLoop(self.X, self.y, theta), 32.07, places=2)
        self.assertAlmostEqual(computeCost(self.X, self.y, theta), 32.07, places=2)
    
    def test_gradientDescent(self):
        theta = np.zeros((2,1))
        iterations = 1500
        alpha = 0.01
        converged_theta = gradientDescent(self.X, self.y, theta, alpha, iterations)
        self.assertAlmostEqual(converged_theta[0], -3.63, places=2)
        self.assertAlmostEqual(converged_theta[1], 1.17, places=2)
        self.assertAlmostEqual(hypothesis(np.array([1,3.5]),converged_theta), 0.45, places=2)
        self.assertAlmostEqual(hypothesis(np.array([1,7]),converged_theta), 4.53, places=2)

if __name__ == '__main__':
    unittest.main()