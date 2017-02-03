load('dataset2.mat');

[e1, e2] = fiveFoldCrossValidation(x, y)

%% 2.1 a)
function [mu, variance] = mle(data)
    mu = mean(data)';
    n = size(data, 1);
    t1 = arrayfun(@(i) sum((data(i,:)' - mu).^2), 1:n);
    variance = sum(t1) / n;
end
%% 2.1 b)
function [P1, P2, Ytest] = sph_bayes(Xtest, x, y)
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

%% 2.1 d)
function [e1, e2]= fiveFoldCrossValidation(x, y)
    CVO = cvpartition(y,'KFold', 5);
    
    err1 = zeros(CVO.NumTestSets);
    err2 = zeros(CVO.NumTestSets);
    for set = 1:CVO.NumTestSets
        trainIdx = CVO.training(set);
        testIdx = CVO.test(set);
               
        trainSetX = x(trainIdx, :);
        trainSetY = y(trainIdx);
        
        testSetX = x(testIdx, :);
        testSetY = y(testIdx, :);
        
        seterr1 = zeros(CVO.TestSize(set), 1);
        seterr2 = zeros(CVO.TestSize(set), 1);
        for test=1:CVO.TestSize(set)
            tmpX = testSetX(test, :);
            tmpY = testSetY(test);
            
            [~, ~, testY1] = sph_bayes(tmpX, trainSetX, trainSetY);
            
            [mu1, ~] = mle(trainSetX(find(trainSetY==1), :));
            [mu2, ~] = mle(trainSetX(find(trainSetY==-1), :));
            testY2 = new_classifier(tmpX', mu1, mu2);
            
            seterr1(test) = (tmpY ~= testY1);
            seterr2(test) = (tmpY ~= testY2);
        end    
        err1(set) = sum(seterr1) / CVO.TestSize(set);
        err2(set) = sum(seterr2) / CVO.TestSize(set);
    end    
    e1 = sum(sum(err1, 1)) / CVO.NumTestSets;
    e2 = sum(sum(err2, 1)) / CVO.NumTestSets;    
end