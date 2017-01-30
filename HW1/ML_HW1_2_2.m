load('dataset1.mat');
[mu, sigma] = sge(x);

mu = mu';
s_range = linspace(0,5, 300);

% Task 2.2 a %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f1 = figure
% values
prior_vals = arrayfun(@(s) prior(s, 1, 1), s_range);
post_vals = arrayfun(@(s) posterior(x, mu, s, 1, 1), s_range);
% plotting
hold on
plot(s_range, prior_vals);
plot(s_range, post_vals);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha = beta = 1)')

f2 = figure
% values
prior_vals = arrayfun(@(s) prior(s, 10, 1), s_range);
post_vals = arrayfun(@(s) posterior(x, mu, s, 10, 1), s_range);
% plotting
hold on
plot(s_range, prior_vals);
plot(s_range, post_vals);
legend('Prior', 'Posterior');
title('Prior and Posterior distributions (alpha = 10; beta = 1)')


saveas(f1,'./hw1_2_2_a1.png')
saveas(f2,'./hw1_2_2_a2.png')

% Task 2.2 b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAP estimates for sigma^2
sMapEstimateAlpha1Beta1 = smap(mu, 1, 1, x)
sMapEstimateAlpha10Beta1 = smap(mu, 10, 1, x)

% Task 2.2 b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model probability
probModelA = modelLikelihood(mu, sMapEstimateAlpha1Beta1, 1, 1, x)
probModelB = modelLikelihood(mu, sMapEstimateAlpha10Beta1, 10, 1, x)

bayesfactor = pm1 / pm2
if bayesfactor > 1
    disp('Model A is preffered over Model B')
else
    disp('Model B is preffered over Model A')
end

function out = modelLikelihood(mu, s, alpha, beta, x)
% P(M|D, mu, s, alpha, beta)
% mu = mean
% s = sigma^2
% x = whole dataset
out = jointLikelihood(x, mu, s) * prior(s, alpha, beta);
end

function out = smap(mu, alpha, beta, x)
% Map estimate for sigma^2
% mu: mean
% x: whole dataset
    tmp = 0;
    for i=1:1:size(x,1)
        xmu = transpose(x(i,:)) - mu;
        tmp = tmp + transpose(xmu) * xmu;
    end
    out = (tmp + 2*beta) / (2*alpha);
end
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
