mu = [1; 1];
sigma = [0.1, -0.05; -0.05, 0.2];

% Generate data from a 2D multivariate normal distribution
rndData = num2cell(transpose(mvnrnd(mu, sigma, 1000)), 1);
% Calculate function values for every cell
values = cellfun(@(x) fun(x, 3), rndData);

% Find values inside (<0) and outside (>0)
inside = cell2mat(rndData(find(values<=0)));
outside = cell2mat(rndData(find(values>0)));

hold on
% plot level sets line for r=1,2,3
ezplot(@(x1, x2) fun([x1;x2], 1), [-1, 3]);
ezplot(@(x1, x2) fun([x1;x2], 2), [-1, 3]);
ezplot(@(x1, x2) fun([x1;x2], 3), [-1, 3]);
% plot generated data points inside and outside of level sets lines
scatter(inside(1,:), inside(2,:), 10, 'blue', 'filled');
scatter(outside(1,:), outside(2,:), 10, 'black', 'filled');

xlabel('x');
ylabel('y');
title(sprintf('Points outside of f(x,3)=0: %i', size(outside, 2)));

function out = fun(x, r)
    mu = [1; 1];
    sigma = [0.1, -0.05; -0.05, 0.2];
    term0 = transpose(minus(x, mu)) * inv(sigma) * minus(x, mu);
    out = term0/2 - r;
end