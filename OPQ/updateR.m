function [R] = updateR(X, R, V, R_iter)
rng(1);
d = size(X, 1);
R = X*V'/(V*V' +0.1*eye(d));
% manifold = euclideanfactory(d, d);
% problem.M = manifold;
% 
% problem.cost  = @(R) norm(X - R * V, 'fro')^2;
% problem.egrad = @(R) -2 * (X - R * V) * V';
% %  -2 * (X - R * V) * V' =0
% 
% % Numerically check gradient consistency (optional).
% checkgradient(problem);
% 
% % Solve.
% % [R, xcost, info, options] = rlbfgs(problem);
% options.maxiter = R_iter;
% options.verbosity = 1;
% % [x, xcost, info, options] = rlbfgs(problem, [], options); 
% [R, xcost, info, options] = steepestdescent(problem, [], options); 
% fprintf('X-UV : now cost %.2f\n', norm(X - R * V, 'fro')^2);
% fprintf('X-UV : now cost %.2f\n', xcost);