h = 3;
d = 2;
N = 5;
D = [1, 3.2, 4.3; 2, 4.2, 6.3];
% X = [1:d*N];
% X = reshape(X, d, N);
X = [1 3 2 1 2;1 3 2 1 2];
B = [1, 3, 2,1,2];
cos_X = zeros(1, N);
D_norm = sqrt(sum(D.*D)); % d * h
X_norm = sqrt(sum(X.*X)); % d * N
DX_norm = zeros(N, 1);
cos_sum = zeros(1, h);
for i=1:N
    cen_of_i = B(i);
    cos_X(i) = 1+dot(D(:,cen_of_i), X(:,i)) / D_norm(cen_of_i)/X_norm(i);
    cos_sum(cen_of_i) = cos_sum(cen_of_i) + cos_X(i);
end
refine_D = zeros(d, h);
for i=1:N
    cen_of_i = B(i);
    refine_D(:, cen_of_i) = refine_D(:, cen_of_i) + cos_X(i)/cos_sum(cen_of_i) * X(:,i);
end
refine_D

