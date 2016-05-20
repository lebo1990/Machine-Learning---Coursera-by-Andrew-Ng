function Costfunction = costfun_annealing(theta)
data = load('ex1data1.txt');
y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

Costfunction = 1 / (2*size(X, 1)) * sum((X * theta - y).^2);