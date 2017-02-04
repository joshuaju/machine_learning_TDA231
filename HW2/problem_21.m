load('dataset2.mat');

[e1, e2] = fiveFoldCrossValidation(x, y)

%% 2.1 d)
function [e1, e2]= fiveFoldCrossValidation(x, y)
% e1 is the error rate for sph_bayes
% e2 is the error rate for new_classifier
    CVO = cvpartition(y,'KFold', 5);    
    % Store error in arrays
    err1 = zeros(CVO.NumTestSets); % error for sph_bayes
    err2 = zeros(CVO.NumTestSets); % error for new_classifier
    for set = 1:CVO.NumTestSets
        trainIdx = CVO.training(set);       
        trainSetX = x(trainIdx, :);
        trainSetY = y(trainIdx);
        
        testIdx = CVO.test(set);
        testSetX = x(testIdx, :);
        testSetY = y(testIdx, :);
        
        seterr1 = zeros(CVO.TestSize(set), 1); % error for sph_bayes per training sret
        seterr2 = zeros(CVO.TestSize(set), 1); % error for new_classifier per training set
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