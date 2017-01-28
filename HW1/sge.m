function [mu, sigma] = sge(x)
%
% SGE Mean and variance estimator for spherical Gaussian distribution                               
%
% x     : Data matrix of size n x p where each row represents a 
%         p-dimensional data point e.g. 
%            x = [2 1;
%                 3 7;
%                 4 5 ] is a dataset having 3 samples each
%                 having two co-ordinates.
%
% mu    : Estimated mean of the dataset [mu_1 mu_2 ... mu_p] 
% sigma : Estimated standard deviation of the dataset (number)                 
%
mu = mean(x);
n = length(x);
sigma = 0;
for i = 1:1:n
    for l = 1:1:2
    sigma = sigma + (x(i,l)-mu(1,l))^2;
    end
end
sigma = sqrt(1/n*sigma);

