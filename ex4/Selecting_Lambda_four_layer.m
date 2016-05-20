%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

% randomly choose training set and cross validation set
xy = [X, y];

%  value to see how more training helps.
options = optimset('MaxIter', 100);

% Lambda
lambda_vec = [0 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2 5 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
pred_train = zeros(length(lambda_vec), 1);
pred_val = zeros(length(lambda_vec), 1);

for i = 1: length(lambda_vec)
    fprintf('\nTraining Neural Network... \n')
    
    xy = [rand(size(X, 1), 1), xy];
    xy = sortrows(xy,1);
    xy = xy(:, 2:end);

    X_train = xy(1: 2500, 1: end - 1);
    X_val = xy(2501 : 5000, 1: end - 1);

    y_train = xy(1: 2500, end);
    y_val = xy(2501 : 5000, end);
    
    %  Initilization   
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
    initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
    % Create "short hand" for the cost function to be minimized
    costFunction_train = @(p) nnCostFunction_four_layer(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda_vec(i));
                                                         
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params_train, ~] = fmincg(costFunction_train, initial_nn_params, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params_train(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params_train(hidden_layer_size * (input_layer_size + 1) + 1 : ...
                        hidden_layer_size * (hidden_layer_size + 1) + hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (hidden_layer_size + 1));

    Theta3 = reshape(nn_params_train(hidden_layer_size * (hidden_layer_size + 1) + hidden_layer_size * (input_layer_size + 1) + 1 : end), ...
                     num_labels, (hidden_layer_size + 1));
    
    nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:)];

	error_train(i) = nnCostFunction_four_layer(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, 0);
	% error_train(i) = linearRegCostFunction(X, y, theta, lambda_vec(i));
	error_val(i) = nnCostFunction_four_layer(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_val, y_val, 0);
	% error_val(i) = linearRegCostFunction(Xval, yval, theta, lambda_vec(i));

    pred_train(i) = mean(double(predict_four_layer(Theta1, Theta2, Theta3, X_train) == y_train)) * 100;
    pred_val(i) = mean(double(predict_four_layer(Theta1, Theta2, Theta3, X_val) == y_val)) * 100;
    
end

% close all;
figure();
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('Lambda \t Traing set prediction \t cross validation set prediction\n');
for i = 1: length(lambda_vec)
    fprintf('%.3f \t', lambda_vec(i));
    fprintf('\t %.3f \t', pred_train(i));
    fprintf('\t \t \t \t %f \t \n', pred_val(i));
end