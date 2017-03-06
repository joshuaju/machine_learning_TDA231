
function [z] = kernelKmeans(data, kernel, K, maxIterations)
    z = kernelAssign(data, kernel, K, maxIterations);    
end

function z = kernelAssign(data, kernel, K, maxIterations)
    N = size(data, 1);
    
    rnd = randi(K, N, 1); % random assignments to K classes
    z = zeros(N, K); % rows refer to data points, columns to class
    for k=1:K
        z(rnd==k, k) = 1;
    end
    converged = 0;
    
    while ~converged && maxIterations ~= 0
        d = kernelDistance(data, kernel, K, z);
        zNew = zeros(N, K);
        for n=1:N
            [~, assignTo] = min(d(n,:));
            zNew(n, assignTo) = 1;
        end
        converged = isequal(z, zNew);
        z = zNew;
        maxIterations = maxIterations - 1;        
    end
end

function d = kernelDistance(data, kernel, K, z)
% d: rows refer to point n, cols are the distance to cluster center
    N = size(data, 1);
    kEval = zeros(N, N); 
    for m=1:N % evaluate kernel-function at all data point combinations
        for l=1:N
            kEval(m,l) = kernel(data(m,:), data(l,:));
        end
    end
    
    Nk = sum(z, 1);    
    term1 = diag(kEval); % kernel(x_n, x_n);    
    term2 = zeros(N, K);
    term3 = zeros(1, K);
    for k=1:K
        zk = z(:,k);               
        term2(:, k) = -2.* Nk(k)^(-1) * sum(repmat(zk',N,1).*kEval,2); % sum(z_mk * k(x_n, x_m)        
        term3(1, k) = Nk(k)^(-2) * sum(sum((zk.*zk') .* kEval)); % sum(z_mk * z_lk * k(x_m, x_l)               
    end
    d = term1 + term2 + term3; % distance per point and cluster    
end

