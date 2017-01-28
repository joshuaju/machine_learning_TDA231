load('dataset1.mat');
[mu, sigma] = sge(x);

mu = mu';
s_range = linspace(0,5, 300);


figure(1)
% values
prior_vals = arrayfun(@(s) prior(s, 1, 1), s_range);
post_vals = arrayfun(@(s) posterior(x, mu, s, 1, 1), s_range);
% plotting
hold on
plot(s_range, prior_vals);
plot(s_range, post_vals);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha = beta = 1)')

figure(2)
% values
prior_vals = arrayfun(@(s) prior(s, 10, 1), s_range);
post_vals = arrayfun(@(s) posterior(x, mu, s, 10, 1), s_range);
% plotting
hold on
plot(s_range, prior_vals);
plot(s_range, post_vals);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha = 10; beta = 1)')


function out = prior(s, a, b)
% The prior distribution (invers-gamma)
% s: sigma^2
% a: alpha
% b: beta
    out = (b.^a) / gamma(a) * s.^(-a-1) * exp(-b/s);
end

function out = likelihood(x, mu, s)
% The likelihood function for a single point x = {x1, x2}
% x : vector
% mu: mean
% s : sigma^2    
    out = 1 / (2*pi*s) * exp(-1 * (transpose(x-mu) * (x-mu)) / (2*s));    
end

function out = jointLikelihood(x, mu, s)
% The likelihood of all points (log-sum-exp trick)
% x: whole dataset
% mu: mean
% s : sigma^2 
    vals = arrayfun(@(i) likelihood(transpose(x(i,:)), mu, s), 1:size(x,1));
    out = log(sum(exp(vals)));
end

function out = posterior(x, mu, s, a, b)
% The posterior distribution
% x: whole dataset
% mu: mean
% s : sigma^2
    vals = jointLikelihood(x, mu, s) * prior(s, a, b);
    out = vals;
end
