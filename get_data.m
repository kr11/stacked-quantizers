function [ X_train, X_test, X_base, gt, nquery ] = get_data( data_set_name, nquery, K)
%     parent_dir = '/Users/kangrong/data/image_dataset/';
    parent_dir = 'data/';
%     parent_dir = '/home/caoyue/kangrong/code/lopq/data/';
    % Load training, query and base datasets, and the query ground truth.
    if strcmp(data_set_name, 'CONVNET_DATASET')
        X = load( 'data/features_m_128.mat', 'feats_m_128_train', 'feats_m_128_test', 'feats_m_128_base');
        X_train = X.feats_m_128_train;
        X_base  = X.feats_m_128_base;
        
        X_test  = X.feats_m_128_test;
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test = X_test(:, 1:nquery);
        
        gt      = load( 'data/features_m_128_gt.mat', 'gt' ); 
        gt      = gt.gt;
        gt      = gt(1:nquery, 1:K)';

    elseif strcmp(data_set_name, 'SIFT_DATASET')
        % Load sift
        X_train = fvecs_read([parent_dir, 'sift/sift_learn.fvecs']);
        X_base  = fvecs_read([parent_dir, 'sift/sift_base.fvecs']);
        
        X_test  = fvecs_read([parent_dir, 'sift/sift_query.fvecs']);
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read([parent_dir, 'sift/sift_groundtruth.ivecs']); 
        gt      = gt';
        gt      = gt(1:nquery, 1:K)'+1;
    elseif strcmp(data_set_name, 'SIFTSMALL_DATASET')
        % Load sift
        X_train = fvecs_read([parent_dir, 'siftsmall/siftsmall_learn.fvecs']);
        X_base  = fvecs_read([parent_dir, 'siftsmall/siftsmall_base.fvecs']);
        
        X_test  = fvecs_read([parent_dir, 'siftsmall/siftsmall_query.fvecs']);
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read([parent_dir, 'siftsmall/siftsmall_groundtruth.ivecs']); 
        gt      = gt';
        gt      = gt(1:nquery, 1:K)'+1;
    elseif strcmp(data_set_name, 'DEEP_DATASET')
        % Load sift
        X_train = fvecs_read([parent_dir, 'deep10M/deep1M_learn.fvecs']);
        X_base  = fvecs_read([parent_dir, 'deep10M/deep1M_base.fvecs']);
        
        X_test  = fvecs_read([parent_dir, 'deep10M/deep1M_query.fvecs']);
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read([parent_dir, 'deep10M/deep1M_groundtruth.ivecs']); 
        gt      = gt';
        gt      = gt(1:nquery, 1:K)'+1;
    elseif strcmp(data_set_name, 'GIST_DATASET')
        % Load gist
        X_train = fvecs_read([parent_dir, 'gist/gist_learn.fvecs']);
        X_base  = fvecs_read([parent_dir, 'gist/gist_base.fvecs']);
        
        X_test  = fvecs_read([parent_dir, 'gist/gist_query.fvecs']);
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read([parent_dir, 'gist/gist_groundtruth.ivecs']); 
        gt      = gt';
        gt      = gt(:, 1:K)'+1;

    end
end

