function w = weights(X, Y, alpha, sv_idx)
    w = sum(alpha .* Y(sv_idx) .* X(sv_idx, :), 1)
end