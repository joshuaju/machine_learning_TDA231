% 2.1 a)
function [mu, variance] = mle(data)
% Compute maximum likelihood estimates
    mu = mean(data)';
    n = size(data, 1);
    t1 = arrayfun(@(i) sum((data(i,:)' - mu).^2), 1:n);
    variance = sum(t1) / n;
end