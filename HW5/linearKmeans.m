function [mu, z] = linearKmeans(data, k, maxIterations)
% data: rows are observations, columns are features
% k: Number of clusters
% maxIterations: Max. iterations (set to 'inf' to keep goind until convergence) 
%
% Return: mu contains cluster centers per columns
% Return: z are the assignment of datapoints (rows) to clusters (cols)

    mu = datasample(data, k)'; % cluster centers in columns    
    lastZ = [-1];
    converged = 0;
    while ~converged && maxIterations ~= 0 % As long as assignments do not change..
        z = assign(data, mu);
        denom = sum(z);
        for tmpK = 1:k % calculate new cluster center for every cluster
            num = sum(data(find(z(:,tmpK) == 1), :));
            mu(:, tmpK) = num / denom(tmpK);
        end
        converged = isequal(lastZ, z); % Did assignments change? 
        lastZ = z;
        maxIterations = maxIterations - 1;
    end
end

function z = assign(data, mu)
% z: Assignment matrix, 
    %row: observation, col: cluster index (1 assigned, 0 not assigned)
% data: each row is an observation
% mu: cluster centers in columns
    z = zeros(size(data, 1), size(mu,1)); % row: observation, col: cluster
    for i=1:size(data, 1)
        tmpX = data(i, :)';
        d = distance(tmpX, mu);
        minIdx = find (d == min(d));
        z(i, minIdx) = 1;
    end
end

function d = distance(x, mu)
%  x: single data point [n x 1]
% mu: cluster center in columns [n x k]
    d = sum((x-mu) .* (x-mu), 1);
end