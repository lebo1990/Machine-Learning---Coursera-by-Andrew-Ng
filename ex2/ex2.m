%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

% plotData(X, y);
% 
% % Put some labels 
% hold on;
% % Labels and Legend
% xlabel('Exam 1 score')
% ylabel('Exam 2 score')
% 
% % Specified in plot order
% legend('Admitted', 'Not admitted')
% hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

   

%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta_fminunc, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta_fminunc);


% Plot Boundary
plotDecisionBoundary(theta_fminunc, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
title('fminunc')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 45 85] * theta_fminunc);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta_fminunc, X);

fprintf('Train Accuracy by Linear Regression (fminunc): %f\n', mean(double(p == y)) * 100);

% pause;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

theta = zeros(n + 1, 1);
alpha = 0.001;
iterations = 500000;
J_history = zeros(iterations, 1);


for iter = 1:iterations
    [cost, grad] = costFunction(theta, X, y);
    theta_new = zeros(n + 1, 1);
    for i = 1: n + 1
        theta_new(i) = theta(i) - alpha * grad(i);
    end
    theta = theta_new;

    % Save the cost J in every iteration    
    J_history(iter) = cost;
end


% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
title(sprintf('gradient @ iterations = %g', iterations))

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled

degree_poly = 2;
X_poly = mapFeature_degree(data(:,1), data(:,2), degree_poly);

[m_poly, n_poly] = size(X_poly);

% Initialize fitting parameters
initial_theta_poly = zeros(n_poly, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta_poly, X_poly, y, lambda);

% Optimize
[theta_poly, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_poly, y, lambda)), initial_theta_poly, options);

% Compute accuracy on our training set
p = predict(theta_poly, X_poly);

fprintf('Train Accuracy by Polynominal Regression: %f\n', mean(double(p == y)) * 100);

% Plot Data
plotData(X(:,2:3), y);
hold on

% Plot Boundary
u = linspace(30, 100, 50);
v = linspace(30, 100, 50);

z = zeros(length(u), length(v));
% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature_degree(u(i), v(j), degree_poly)*theta_poly;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
hold off;
title(sprintf('polynominal regression @ degree = %g', degree_poly))

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend('Admitted', 'Not admitted')
hold off;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

figure();
plot(J_history);
title('Cost Function');
% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy by Linear Regression (Gradient): %f\n', mean(double(p == y)) * 100);