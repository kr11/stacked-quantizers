function [RX, model] = coc_reorder( RX, model, m)
    
    % kmeans for each dimension
    [idx, ~] = kmeans(RX, m, 'Start', 'plus');
    coc_idx = cell(m:1);
    for i=1:m,
        coc_idx{i} = find(idx == i);
    end
    %reordering RX
    RX = RX(cat(1, coc_idx{:}),:);
    cluster_sizes = cellfun(@(x)size(x,1), coc_idx);
    len0 = 1 + cumsum([0 cluster_sizes(1:end-1)]);
    len1 = cumsum(cluster_sizes);
    model.coc_idx = coc_idx;
    model.len0 = len0';
    model.len1 = len1';

end

