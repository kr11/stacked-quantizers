function [Q, RX] = quantize_by_ckmeans(X, model, use_coc)

% Inputs:
%       X: p*n -- n p-dimensional input points to be quantized.
%       model: a model created by ckmeans function -- it should
%              contain: R, centers, m, h, len

% Outputs:
%       Q: m*n -- n m-dim subcenter assignment vectors. NOTE: each
%          entry is betwen 0 and h-1. For now we assume h <= 256,
%          so each entry of Q is uint8.
%       RX: q*n -- transformation of the data-points into a space in
%           which the subspaces are defined. NOTE: q might be smaller
%           than p if ckmeans is run after PCA pre-processing.

p = size(X, 1);
n = size(X, 2);
m = model.m;
Q = zeros (m, n, 'int32');

if any(model.h > 256)
  error('uint8 indices required');
end
% len0 = [1 33 65 97]' len1 = [32 64 96 128]'
len0 = 1 + cumsum([0; model.len(1:end-1)]);
len1 = cumsum(model.len);
RX = model.R \ X;

if use_coc
    len0 = model.len0;
    len1 = model.len1;
    RX = RX(cat(1, model.coc_idx{:}),:);
%     RX = model.R \ X_coc;
end

for i = 1:m
  % find the nearest center for each subvector after rotation.
  ind = euc_nn_mex(model.centers{i}, RX(len0(i):len1(i), :));
  Q(i, :) = int32(ind');  % zero-based
end
