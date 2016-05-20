function [J, theta] = momentum_adaptive(theta, X, y, alpha, beta, lambda, iterations)
	input_layer_size = size(theta, 1);
	num_labels = size(theta, 2);

	J = zeros((iterations + 1), 1);
    
    grad = zeros(size(theta));
    grad_tmp = zeros(size(theta));

	g_limit = [0.1, 10];
    g = ones(size(theta));
    
    theta_delta = zeros(size(theta));

	

	for i = 1: iterations + 1
		% cal
		[J(i), grad_new] = nnCostFunction_two_layer2(theta, X, y, lambda);
		g1 = bsxfun(@gt, grad, zeros(size(grad)));
		g2 = bsxfun(@gt, grad_tmp, zeros(size(grad_tmp)));
		t = 1-bsxfun(@xor, g1, g2);
		g_new = g .* (0.05 * t + 0.95) + 0.05 * t;
		for p = 1: size(g_new, 1)
			for q = 1: size(g_new, 2)
				if g_new(p, q) > g_limit(2)
					g_new(p, q) = g_limit(2);
				elseif g_new(p, q) < g_limit(1)
					g_new(p, q) = g_limit(1);
				end
			end
        end
        
		theta_delta_new = beta * theta_delta - alpha * g_new .* grad_new;
		theta_new = theta + theta_delta;
		% update
		grad_tmp = grad;
		grad = grad_new;
		g_tmp = g;
		g = g_new;
		theta_tmp = theta;
		theta = theta_new;
		theta_delta_tmp = theta_delta;
		theta_delta = theta_delta_new;	
	end
    