load('dataset1.mat');
[mu, sigma] = sge(x);

mu = mu';
s_range = linspace(0, 1, 500);

% Task a ------------------------------------------------------------------

% Plot for alpha = 1 and beta = 1
alpha = 1; beta = 1;
prior_values = arrayfun(@(s) prior(s, alpha, beta), s_range);
posterior_values = arrayfun(@(s) posterior(mu, s, alpha, beta, x), s_range);

f1 = figure;
hold on
plot(s_range, prior_values);
plot(s_range, posterior_values);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha=beta=1)')
xlabel('\sigma^2')

% Plot for alpha = 10 and beta = 1
alpha = 10; beta = 1;
prior_values = arrayfun(@(s) prior(s, alpha, beta), s_range);
posterior_values = arrayfun(@(s) posterior(mu, s, alpha, beta, x), s_range);

f2 = figure;
hold on
plot(s_range, prior_values);
plot(s_range, posterior_values);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha=10; beta=1)')
xlabel('\sigma^2')

% Task b ------------------------------------------------------------------
modelA_map = map(mu, 1, 1, x)
modelB_map = map(mu, 10, 1, x)

% Task c ------------------------------------------------------------------
posteriorA = posterior(mu, modelA_map, 1, 1, x)
posteriorB = posterior(mu, modelB_map, 10, 1, x)
bayesFactor = posteriorA / posteriorB

if bayesFactor > 1
    disp('Model A is preffered over Model B')
else
    disp('Model B is preffered over Model A')
end

function out = prior(s, a, b)
    out = exp(a .* log(b)- sum(log([1:a-1]))+ (-a-1).* log(s)- b./s);
end

function out = posterior(mu, s, a, b, x)
    [an, bn] = hyperparameter(mu, a, b, x);    
    out = prior(s, an, bn);
end

function out = map(mu, a, b, x)
    [an, bn] = hyperparameter(mu, a, b, x);
    out = bn / (an + 1);
end

function [an, bn] = hyperparameter(mu, a, b, x)
    an = a + size(x, 1);
    tmp = arrayfun(@(i) transpose((transpose(x(i,:))-mu))*(transpose(x(i,:))-mu), 1:size(x,1));
    bn = sum(tmp)/2 + b;    
end