% 2.1 b)
function [P1, P2, Ytest] = sph_bayes(Xtest, x, y)
% P1 likelihood of point Xtest to belong to class 1
% P2 likelihood of point Xtest to belong to class -1
% YTest is the determined label 
    [mu_Pos1, var_Pos1] = mle( x(find(y == 1) , :));
    [mu_Neg1, var_Neg1] = mle( x(find(y == -1), :));
    
    P1 = sphGaussian(Xtest, mu_Pos1, var_Pos1);
    P2 = sphGaussian(Xtest, mu_Neg1, var_Neg1);
    
    P1 = P1 / (P1 + P2);
    P2 = P2 / (P1 + P2);
    
    if P1 > P2
        Ytest = 1;
    elseif P1 < P2
        Ytest = -1;
    else
        Ytest = 0;
    end
end
function out = sphGaussian(point, mu, s)
    t1 = 1 / sqrt( 2* pi^3 * s^3);
    t2 = exp(-1 * sum((point'-mu).^2) / (2*s));
    out = t1*t2;
end