% 2.1 c)
function [YTest] = new_classifier(Xtest, mu1, mu2)
    b = 0.5 * (mu1 + mu2); % 1 1100
    Xtest-b;
    nom = (mu1-mu2)' * (Xtest-b); % (1100 1)' * (512 1100 - 1 1100) 
    denom = norm( mu1-mu2 );
    YTest = sign(nom/denom);
end  

