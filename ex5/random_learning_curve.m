%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 8;

% % Map X onto Polynomial Features and Normalize
% X_poly = polyFeatures(X, p);
% [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
% X_poly = [ones(m, 1), X_poly];                   % Add Ones
% 
% % Map X_poly_test and normalize (using mu and sigma)
% X_poly_test = polyFeatures(Xtest, p);
% X_poly_test = bsxfun(@minus, X_poly_test, mu);
% X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
% X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
% 
% % Map X_poly_val and normalize (using mu and sigma)
% X_poly_val = polyFeatures(Xval, p);
% X_poly_val = bsxfun(@minus, X_poly_val, mu);
% X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
% X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones
% 
% fprintf('Normalized Training Example 1:\n');
% fprintf('  %f  \n', X_poly(1, :));
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% % pause;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
lambda = 0.01;
num = 12;
p = 8;
iterations = 5;
% 
% lambda = 3;
% [theta] = trainLinearReg(X_poly, y, lambda);
% 
% 
% figure();
% [error_train, error_val] = ...
%     learningCurve(X_poly, y, X_poly_val, yval, lambda);
% plot(1:m, error_train, 1:m, error_val);
% 
% title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
% xlabel('Number of training examples')
% ylabel('Error')
% axis([0 13 0 100])
% legend('Train', 'Cross Validation')
% 
% fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:m
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end

X_total = [X; Xval;Xtest];
y_total = [y; yval;ytest];

figure();
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5); hold on;
plot(Xval, yval, 'o', 'MarkerSize', 10, 'LineWidth', 1.5); hold on;
plot(Xtest, ytest, 'k.', 'MarkerSize', 10, 'LineWidth', 1.5); hold on;
legend('Train',' Cross Validation','Test');
% Map X onto Polynomial Features and Normalize
X_total_poly = polyFeatures(X_total, p);
[X_total_poly, ~, ~] = featureNormalize(X_total_poly);  % Normalize
X_total_poly = [ones(size(X_total_poly, 1), 1), X_total_poly];  % Add Ones
Xy_total_poly = [X_total_poly, y_total];                   

Xtrain_12 = Xy_total_poly(1: num, 1: end - 1);
ytrain_12 = Xy_total_poly(1: num, end);

Xy_val_total = Xy_total_poly(num + 1: end, :);

error_train_mean = zeros(iterations, num);
error_val_mean = zeros(iterations, num);

for t = 1:iterations
%     Xy_val_total = [rand(size(Xy_val_total, 1), 1), Xy_val_total];
%     Xy_val_total = sortrows(Xy_val_total,1);
%     Xy_val_total = Xy_val_total(:, 2:end);
%     Xval_12 = Xy_val_total(1: num, 1: end - 1);
%     yval_12 = Xy_val_total(1: num, end);

    randidx = randperm(size(Xy_val_total, 1)); %shuffle
    
    Xval_12 = Xy_val_total(randidx(1:num), 1: end - 1);
    yval_12 = Xy_val_total(randidx(1:num), end);


    
    theta = trainLinearReg(Xtrain_12, ytrain_12, lambda);
    [error_train_mean(t, :), error_val_mean(t, :)] = learningCurve(Xtrain_12, ytrain_12, Xval_12, yval_12, lambda);
end

error_train_mean_result = mean(error_train_mean, 1);
error_val_mean_result = mean(error_val_mean, 1);

figure();
plot(1:num, error_train_mean_result, 1:num, error_val_mean_result);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

% fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:m
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end