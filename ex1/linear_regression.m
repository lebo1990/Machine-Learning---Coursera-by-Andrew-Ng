clear all; close all; clc

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples


X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1000;
alpha = 0.01;
J_history = zeros(iterations, 1);

[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

subplot(2,1,2)
plot(J_history);

% final:
% target_theta = [-3.6303;1.1664];
% X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% hold on;plot(X(:,2), X*target_theta, '-')

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
