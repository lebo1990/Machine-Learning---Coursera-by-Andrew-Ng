function [theta, J_history, g0, g1] = gradientDescent_adaptive_learning_rates(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

limit = [0.1, 10];

g0 = [1; zeros(num_iters, 1)];
g1 = [1; zeros(num_iters, 1)];

grad0_tmp = 1;
grad1_tmp = 1;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
    htheta = X * theta;
    grad0 = sum((htheta - y) .* X(:,1));
    grad1 = sum((htheta - y) .* X(:,2));
         
    switch grad0_tmp * grad0 > 0
        case 1
            g0(iter + 1) = g0(iter) + 0.05;
        case 0
            g0(iter + 1) = g0(iter) * 0.95;
        otherwise
            g0(iter + 1) = g0(iter) * 0.95;
    end
    switch grad1_tmp * grad1 > 0
        case 1
            g1(iter + 1) = g1(iter) + 0.05;
        case 0
            g1(iter + 1) = g1(iter) * 0.95;
        otherwise
            g1(iter + 1) = g1(iter) * 0.95;
    end

    if g0(iter + 1) > limit(2)
        g0(iter + 1) = limit(2);
    elseif g0(iter + 1) < limit(1)
        g0(iter + 1) = limit(1);
    end
    if g1(iter + 1) > limit(2)
        g1(iter + 1) = limit(2);
    elseif g1(iter + 1) < limit(1)
        g1(iter + 1) = limit(1);
    end
    
    
    theta0 = theta(1) - alpha * g0(iter + 1) / m * grad0;
    theta1 = theta(2) - alpha * g1(iter + 1) / m * grad1; 
    theta = [theta0; theta1];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
