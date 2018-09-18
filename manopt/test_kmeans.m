% load fisheriris
% X = meas(:,3:4);
% figure;
% plot(X(:,1),X(:,2),'k*','MarkerSize',5);
% title 'Fisher''s Iris Data';
% xlabel 'Petal Lengths (cm)'; 
% ylabel 'Petal Widths (cm)';
% rng(1); % For reproducibility
% [idx,C] = kmeans(X,3);

X_train = fvecs_read('/home/caoyue/kangrong/code/lopq/data/sift/sift_learn.fvecs');
% X_train = fvecs_read('/home/caoyue/kangrong/code/lopq/data/deep10M/deep1M_train.fvecs');
%X_train = X_train;
X_train = bsxfun(@rdivide, X_train, sum(X_train.^2, 2).^0.5); 
[idx,C] = kmeans(X_train, 4, 'Start', 'plus');