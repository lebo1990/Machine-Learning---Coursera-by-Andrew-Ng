% options = optimoptions(@simulannealbnd,'MaxIterations',1000);
% options.MaxIter = 1e3;
lb = -inf(2,1);
ub = inf(2,1);

ObjectiveFunction = @ costfun_annealing;
theta = zeros(2, 1); % initialize fitting parameters
[x,fval,exitFlag,output] = simulannealbnd(ObjectiveFunction, theta, lb, ub)