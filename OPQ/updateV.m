function [V] = updateV(X, R, V, CB, lambda, V_iter)
rng(1);
d = size(X, 1);
n = size(X, 2);
V = (R'* R + lambda * eye(d)) \ (R' * X + lambda * CB);
% manifold = euclideanfactory(d, n);
% problem.M = manifold;
% 
% problem.cost  = @(V) norm(X - R * V, 'fro')^2 + lambda * norm(V - CB, 'fro')^2;
% problem.egrad = @(V) 2 * lambda * (V - CB) - 2 * R' * (X - R * V);
% 
% % Numerically check gradient consistency (optional).
% checkgradient(problem);
%  
% % Solve.
% % [V, xcost, info, options] = rlbfgs(problem);
% options.maxiter = V_iter; 
% options.verbosity = 1;
% [V, xcost, info, options] = rlbfgs(problem, [], options); 
% % [V, xcost, info, options] = steepestdescent(problem, [], options); 
% 
% % fprintf('now cost %.2f\n', norm(X - R * V, 'fro')^2 + lambda * norm(V - CB, 'fro')^2);
% fprintf('|X-UV|^2 + lambda * |V-CB|^2 : now cost %.2f\n', xcost);
