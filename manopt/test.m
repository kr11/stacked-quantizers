% Generate random problem data.
% n = 5000;
% k = 8;
% d = 64;
k = 3;
d = 2;
load fisheriris
X = meas(:,3:4);
n = size(X,1);

rng(1);
[idx,C] = kmeans(X, k);
% X = randn(d, n);
B = zeros(k, n);
for i = 1:n
%     B(unidrnd(k),i) = 1;
    B(idx(i),i) = 1;
end


B = B';
o_cost_fun = norm(X - B * C, 'fro')^2;

% original cost function

% Create the prblem structure.
manifold = stiefelfactory(k, d);
problem.M = manifold;
 
% Define the problem cost function and its Euclidean gradient.
% problem.cost  = @(C) norm(X-C*B, 'fro')^2;
% problem.egrad = @(C) -2 * (X-C*B) * B';
problem.cost  = @(C) norm(X - B * C, 'fro')^2;
problem.egrad = @(C) -2 * B' *(X - B * C);

% Numerically check gradient consistency (optional).
checkgradient(problem);
 
% Solve.
options.maxiter = 100;
options.verbosity = 2;
[x, xcost, info, options] = rlbfgs(problem,[],options);

% [x, xcost, info, options] = (problem);
 
% Display some statistics.
% figure;
% semilogy([info.iter], [info.gradnorm], '.-');
% xlabel('Iteration number');
% ylabel('Norm of the gradient of f');
% a = x' * x;
fprintf('original cost %.2f\n', o_cost_fun);
fprintf('now cost %.2f\n', norm(X - B * x, 'fro')^2);

% draw
% figure;
% plot(X(:,1),X(:,2),'k*','MarkerSize',5);
% title 'Fisher''s Iris Data';
% xlabel 'Petal Lengths (cm)'; 
% ylabel 'Petal Widths (cm)';

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)

plot(x(:,1),x(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off