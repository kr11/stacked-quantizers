function [] = print_recall( recall_at_k )
    fprintf('%.3f %.3f %.3f %.3f %.3f\n', recall_at_k(1), ...
        recall_at_k(10), recall_at_k(100), ...
        recall_at_k(1000), recall_at_k(10000));

end

