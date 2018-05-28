% Approximate nearest neighbour search demo.

% --
% Julieta, 2014

clc
clear
clf
addpath(genpath(pwd));


% data_set_name = 'CONVNET_DATASET';
% data_set_name = 'SIFT_DATASET';
data_set_name = 'DEEP_DATASET';
% data_set_name = 'GIST_DATASET';

nquery = 10000;
K = 1;  % K is the number of nearest neigher
[ X_train, X_test, X_base, gt, nquery ] = get_data( data_set_name, nquery, K);

% execute algorithm
PQ_EXE = 1;
PQ_RK_EXE = 1;
PQ_COC_EXE = 0;
OPQ_EXE = 0;
OPQ_COC_EXE = 0;
RPQ_EXE = 0;
RPQ_COC_EXE = 0;
SQ_EXE = 0;
AQ_EXE = 0;

%% Set train and search parameters
m       = 4;   % Number of subcodebooks.
h       = 256; % Number of cluster centres per subcodebook.
nbits   = log2(h) * m;  % Number of bits in the final code.

% === The following values are here for a quick demo. 
nitsOPQ = 10; % Iterations in PQ / OPQ. (100 in paper)
% nitsRPQ = 10; % Iterations in PQ / OPQ. (100 in paper)
nitsSQ  = 10; % Iterations in SQ.       (100 in paper)
nitsAQ  = 10; % Iterations in AQ.       ( 10 in paper)

verbose = 1; % Print progress.

N_train = 16; % Pool size for beam search in AQ during training.
N_base  = 64; % Pool size for beam search in AQ during database encoding.

selectivity = 10000; % Number of nearest-neighbours to retrieve.

% semilogx( zeros(selectivity, 1), 'b-', 'linewidth', 0.1 );
    
%% === PQ (no preprocessing) ===
if PQ_EXE == 1
    fprintf('=== PQ: %d codebooks. ===\n', m);

    % Train
    [model, ~] = product_quantization( X_train, m, h, nitsOPQ );

    % Quantize the database
    cbase = uint8( quantize_by_ckmeans(X_base, model, false) -1 );

    % Search
    centers = double(cat(1, model.centers{:}));
    npoints = size(cbase, 2);

    fprintf('Searching... '); tic;
    queryR       = double( model.R' * X_test );
    [ids_aqd, ~] = linscan_aqd_knn_mex(cbase, queryR, npoints, nbits, selectivity, centers, int32(model.len1));
    fprintf('done in %.2f seconds\n', toc);

    % Plot recall@N curve
    recall_at_k_aqd_pq = eval_recall_vs_sel( double(ids_aqd'), nquery, double(gt'), K, selectivity );
    semilogx( recall_at_k_aqd_pq, 'b-', 'linewidth', 2 ); 
    grid on; hold on; xlabel('N'); ylabel('Recall@N');
    legend('PQ', 'location', 'northwest');
    pause(0.5);
end
%% === PQ RK (no preprocessing) ===
if PQ_RK_EXE == 1
    fprintf('=== PQ: %d codebooks. ===\n', m);

    % Train
    [model, ~] = product_quantization_rkmeans( X_train, m, h, nitsOPQ );

    % Quantize the database
    cbase = uint8( quantize_by_ckmeans(X_base, model, false) -1 );

    % Search
    centers = double(cat(1, model.centers{:}));
    npoints = size(cbase, 2);

    fprintf('Searching... '); tic;
    queryR       = double( model.R' * X_test );
    [ids_aqd, ~] = linscan_aqd_knn_mex(cbase, queryR, npoints, nbits, selectivity, centers, int32(model.len1));
    fprintf('done in %.2f seconds\n', toc);

    % Plot recall@N curve
    recall_at_k_aqd_pq_rk = eval_recall_vs_sel( double(ids_aqd'), nquery, double(gt'), K, selectivity );
    semilogx( recall_at_k_aqd_pq_rk, 'm-', 'linewidth', 2 ); 
    grid on; hold on; xlabel('N'); ylabel('Recall@N');
    legend('PQ', 'PQ\_RK', 'location', 'northwest');
    pause(0.5);
end
%% === PQ_COC (no preprocessing) ===
if PQ_COC_EXE == 1
    fprintf('=== PQ_COC: %d codebooks. ===\n', m);

    % Train
    [model_coc, ~] = product_quantization_coc( X_train, m, h, nitsOPQ );
    % Quantize the database
    cbase_coc = uint8( quantize_by_ckmeans(X_base, model_coc, true) -1 );

    % Search
    centers_coc = double(cat(1, model_coc.centers{:}));
    npoints_coc = size(cbase_coc, 2);

    fprintf('Searching... '); tic;
    queryR_coc       = double( model_coc.R' * X_test );
    queryR_coc = queryR_coc(cat(1, model_coc.coc_idx{:}),:);

    
    [ids_aqd_coc, ~] = linscan_aqd_knn_mex( cbase_coc, queryR_coc, npoints_coc, nbits, selectivity, centers_coc, int32(model_coc.len1));
    fprintf('done in %.2f seconds\n', toc);

    % calculate recall
    recall_at_k_aqd_pq_coc = eval_recall_vs_sel( double(ids_aqd_coc'), nquery, double(gt'), K, selectivity );

    % Plot recall@N curve
    print_recall(recall_at_k_aqd_pq)
    print_recall(recall_at_k_aqd_pq_coc)
    model_coc.len1 - model_coc.len0 + 1
    
    semilogx( recall_at_k_aqd_pq, 'b-', 'linewidth', 2 ); 
    legend('PQ', 'location', 'northwest');
    grid on; hold on; xlabel('N'); ylabel('Recall@N');
    semilogx( recall_at_k_aqd_pq_coc, 'g-', 'linewidth', 2 ); 
    legend('PQ', 'PQ_COC', 'location', 'northwest');
    pause(0.5);
end
%% === OPQ ===
if OPQ_EXE == 1
    fprintf('=== OPQ: %d codebooks. ===\n', m);

    % [idx,C] = kmeans(X_train, m, 'Start', 'plus');

    % Train
    [model_OPQ, ~] = ckmeans(X_train, m, h, nitsOPQ, 'natural', false);
    % Quantize the database
    cbase_OPQ = uint8( quantize_by_ckmeans(X_base, model_OPQ, false) -1 );

    % Search
    centers_OPQ = double(cat(1, model_OPQ.centers{:}));
    npoints_OPQ = size(cbase_OPQ, 2);

    fprintf('Searching... '); tic;
    queryR_OPQ       = double( model_OPQ.R' * X_test );
    [ids_aqd_OPQ, ~] = linscan_aqd_knn_mex( cbase_OPQ, queryR_OPQ, npoints_OPQ, nbits, selectivity, centers_OPQ, int32(model_OPQ.len1));

    fprintf('done in %.2f seconds\n', toc);

    
    % Plot recall@N curve
    recall_at_k_aqd_OPQ = eval_recall_vs_sel( double(ids_aqd_OPQ'), nquery, double(gt'), K, selectivity );
    array_len0= model_OPQ.len0';
    array_recall = recall_at_k_aqd_OPQ;

    semilogx( recall_at_k_aqd_OPQ, 'r-', 'linewidth', 2 );
    grid on; hold on; xlabel('N'); ylabel('Recall@N');

%     legend('OPQ', 'location', 'northwest');
    pause(0.5);
end

%% === OPQ_COC ===
if OPQ_COC_EXE == 1
    for i=1:1
        fprintf('iter:%d\n', i);
        fprintf('=== OPQ: %d codebooks. ===\n', m);

        % [idx,C] = kmeans(X_train, m, 'Start', 'plus');

        % Train
        [model_OPQ_COC, ~] = ckmeans(X_train, m, h, nitsOPQ, 'natural', true);
        % Quantize the database
        cbase_OPQ_COC = uint8( quantize_by_ckmeans(X_base, model_OPQ_COC,true) -1 );

        % Search
        centers_OPQ_COC = double(cat(1, model_OPQ_COC.centers{:}));
        npoints_OPQ_COC = size(cbase_OPQ_COC, 2);

        fprintf('Searching... '); tic;
        queryR_OPQ_COC       = double( model_OPQ_COC.R' * X_test );
        queryR_OPQ_COC       =  queryR_OPQ_COC(cat(1, model_OPQ_COC.coc_idx{:}),:);

        [ids_aqd_OPQ_COC, ~] = linscan_aqd_knn_mex( cbase_OPQ_COC, queryR_OPQ_COC, npoints_OPQ_COC, nbits, selectivity, centers_OPQ_COC, int32(model_OPQ_COC.len1));
        fprintf('done in %.2f seconds\n', toc);

        recall_at_k_aqd_OPQ_COC = eval_recall_vs_sel( double(ids_aqd_OPQ_COC'), nquery, double(gt'), K, selectivity );

        array_len0 = [array_len0; model_OPQ_COC.len0'];
        array_recall = [array_recall; recall_at_k_aqd_OPQ_COC];

        % Plot recall@N curve
    %     semilogx( recall_at_k_aqd_OPQ, 'r-', 'linewidth', 2 );
    %     grid on; hold on; xlabel('N'); ylabel('Recall@N');
        semilogx( recall_at_k_aqd_OPQ_COC, 'g-', 'linewidth', 2 );
    %     legend('OPQ_COC', 'location', 'northwest');
        pause(0.5);
    end
end

%% === RPQ ===
if RPQ_EXE == 1
    fprintf('=== RPQ: %d codebooks. ===\n', m);
    % Train
    nitsRPQ = 10;
    lambda = 20;
    R_iter = 1200;
    V_iter = 50;
    [model, ~] = r_ckmeans(X_train, m, h, nitsRPQ, 'OPQ', lambda, R_iter, V_iter, model_OPQ);

    % Quantize the database
    cbase = uint8( quantize_by_ckmeans(X_base, model) -1 );

    % Search
    centers = double(cat(3, model.centers{:}));
    npoints = size(cbase, 2);

    fprintf('Searching... '); tic;
    queryR       = double( model.R \ X_test );
    [ids_aqd, ~] = linscan_aqd_knn_mex( cbase, queryR, npoints, nbits, selectivity, centers);
    fprintf('done in %.2f seconds\n', toc);

    % Plot recall@N curve
    recall_at_k_aqd_rpq = eval_recall_vs_sel( double(ids_aqd'), nquery, double(gt'), K, selectivity );

    semilogx( recall_at_k_aqd_pq, 'b-', 'linewidth', 2 ); 
    grid on; hold on; xlabel('N'); ylabel('Recall@N');
    legend('PQ', 'location', 'northwest');
    semilogx( recall_at_k_aqd_opq, 'r-', 'linewidth', 2 );
    legend('PQ', 'OPQ', 'location', 'northwest');
    semilogx( recall_at_k_aqd_rpq, 'g-', 'linewidth', 2 );
    legend('PQ', 'OPQ', 'RPQ', 'location', 'northwest');
    pause(0.5);
end
 
%% === SQ ===
if SQ_EXE == 1
    fprintf('=== SQ ncodebooks %d ===.\n', m);

    % Train
    [~, codebooks] = SQ_pipeline( X_train, h, m, nitsSQ, verbose );

    % Quantize the database
    cbase = SQ_encode( X_base, codebooks, verbose );

    % Compute database l2 norms.
    dbnorms = single( sum( SQ_decode( cbase,  codebooks ).^2, 1 ) );

    % Convert cbase to uint8
    cbase = uint8( cbase -1 );

    fprintf('Searching... '); tic;
    [~, idx] = SQ_search( cbase, codebooks, X_test, dbnorms, selectivity);
    fprintf('done in %.2f seconds\n', toc);

    % Plot recall@N curve
    recall_at_k_aqd_sq = eval_recall_vs_sel( double(idx'), nquery, double(gt'), K, 10000 );
    semilogx( recall_at_k_aqd_sq, 'm-', 'linewidth', 2 );
    legend('PQ', 'OPQ', 'SQ', 'location', 'northwest');
    pause(15);
end

if AQ_EXE == 1
    %% === AQ === 
    fprintf('=== AQ ncodebooks %d ===.\n', m);

    % Train
    [codes, codebooks, distortions, R] = AQ_pipeline( X_train, h(ones(1,m)), nitsAQ, N_train, verbose );

    % Quantize the database
    fprintf('Encoding the database; this is gonna take a while... '); tic;
    cbase = AQ_encoding( R'*X_base, codebooks, N_base, verbose );
    fprintf('done in %.2f seconds\n', toc);

    % Compute database l2 norms.
    dbnorms = single( sum( SQ_decode( cbase,  codebooks ).^2, 1 ) );

    % Convert cbase to uint8
    cbase = uint8( cbase -1 );

    fprintf('Searching... '); tic;
    for i = 1:numel(codebooks), codebooks{i} = single( codebooks{i} ); end
    [~, idx] = SQ_search( cbase, codebooks, X_test, dbnorms, selectivity);
    fprintf('done in %.2f seconds\n', toc);

    % Plot recall@N curve
    recall_at_k_aqd_aq = eval_recall_vs_sel( double(idx'), nquery, double(gt'), K, 10000 );
    semilogx( recall_at_k_aqd_aq, 'k-', 'linewidth', 2 );
    legend('PQ', 'OPQ', 'SQ', 'AQ', 'location', 'northwest');
end
plan_name = 'refine_kmeans';
title([plan_name, '-', data_set_name]);
saveas(gcf,[plan_name, '_', data_set_name, '.jpg'])
