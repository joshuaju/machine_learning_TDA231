load('dataset2.mat');

%[mu1, ~] = mle(x(find(y==1), :));
%[mu2, ~] = mle(x(find(y==-1), :));
%vals = arrayfun(@(i) new_classifier(x(i,:)', mu1, mu2), 1:size(x,1));


%% 2.1 a)
function [mu, variance] = mle(data)
    mu = mean(data)';
    n = size(data, 1);
    t1 = arrayfun(@(i) sum((data(i,:)' - mu).^2), 1:n);
    variance = sum(t1) / n;
end

function [Ytest] = sph_bayes(Xtest, x, y)
    [mu_Pos1, var_Pos1] = mle( x(find(y == 1) , :));
    [mu_Neg1, var_Neg1] = mle( x(find(y == -1), :));
    
    P1 = sphGaussian(Xtest, mu_Pos1, var_Pos1);
    P2 = sphGaussian(Xtest, mu_Neg1, var_Neg1);
    if P1 > P2
        Ytest = 1;
    elseif P1 < P2
        Ytest = -1;
    else
        Ytest = 0;
    end
end
%% 2.1 b)
function out = sphGaussian(point, mu, s)
    t1 = 1 / sqrt( 2* pi^3 * s^3);
    t2 = exp(-1 * sum((point-mu).^2) / (2*s));
    out = t1*t2;
end  
%% 2.1 c)
function [YTest] = new_classifier(Xtest, mu1, mu2)
    b = 0.5 * (mu1 + mu2);
    nom = (mu1-mu2)' * (Xtest-b);
    denom = sqrt( sum( (mu1-mu2).^2 ) );
    YTest = sign(nom/denom);
end  