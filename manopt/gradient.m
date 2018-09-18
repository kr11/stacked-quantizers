% Generate random problem data.
% n = 5000;
% k = 8;
% d = 64;
k = 64;
d = 96;
n = 10000;

X = fvecs_read('/Users/kangrong/data/image_dataset/deep10M/deep10M.fvecs',n);
% X: d * n
rng(1);
[idx,C] = kmeans(X', k); % C: k * d
C = C'; % C: d * k 
B = zeros(k, n);
for i = 1:n
%     B(unidrnd(k),i) = 1;
    B(idx(i),i) = 1;
end
o_cost_fun = norm(X - C * B, 'fro')^2;
fprintf('diff with I: %.2f\n', norm(C * C' - eye(d), 'fro')^2);

% a = 10;
% for no = 1:10
%     for it = 1:10
%           %C = X*B'/(B*B' + 200* C'*C-100);
%           C = (X*B' + 2*a * C - 2 * a * (C* C') * C)/(B*B');
%     end
%     %fprintf('diff with I: %.2f\n', norm(C' * C - eye(k), 'fro')^2);
%     fprintf('diff with I: %.2f\n', norm(X - C * B, 'fro')^2);
% end
