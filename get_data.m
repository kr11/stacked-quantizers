function [ X_train, X_test, X_base, gt, nquery ] = get_data( data_set_name, nquery, K)
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
        X_train = fvecs_read('/home/caoyue/kangrong/code/lopq/data/sift/sift_learn.fvecs');
        X_base  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/sift/sift_base.fvecs');
        
        X_test  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/sift/sift_query.fvecs');
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read('/home/caoyue/kangrong/code/lopq/data/sift/sift_groundtruth.ivecs'); 
        gt      = gt';
        gt      = gt(1:nquery, 1:K)'+1;
    elseif strcmp(data_set_name, 'DEEP_DATASET')
        % Load sift
        X_train = fvecs_read('/home/caoyue/kangrong/code/lopq/data/deep10M/deep1M_learn.fvecs');
        X_base  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/deep10M/deep1M_base.fvecs');
        
        X_test  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/deep10M/deep1M_query.fvecs');
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read('/home/caoyue/kangrong/code/lopq/data/deep10M/deep1M_groundtruth.ivecs'); 
        gt      = gt';
        gt      = gt(1:nquery, 1:K)'+1;
    elseif strcmp(data_set_name, 'GIST_DATASET')
        % Load gist
        X_train = fvecs_read('/home/caoyue/kangrong/code/lopq/data/gist/gist_learn.fvecs');
        X_base  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/gist/gist_base.fvecs');
        
        X_test  = fvecs_read('/home/caoyue/kangrong/code/lopq/data/gist/gist_query.fvecs');
        nquery  = min(size( X_test, 2 ), nquery); % Number of query vectors.
        X_test  = X_test(:, 1:nquery);
        
        gt      = ivecs_read('/home/caoyue/kangrong/code/lopq/data/gist/gist_groundtruth.ivecs'); 
        gt      = gt';
        gt      = gt(:, 1:K)'+1;

    end
end

