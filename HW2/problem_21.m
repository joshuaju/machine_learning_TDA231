load('dataset2.mat');

nfolds = 5;
nObservations = length(y);

indicies = crossvalind('Kfold', nObservations, nfolds);

% error matrices: rows correspond to fold, columns to test points
err1 = zeros(nfolds, 1);
err2 = zeros(nfolds, 1);
for tmpFold=1:nfolds
    test = (indicies == tmpFold); train = ~test;
    
    xtrain = x(train==1, :);
    ytrain = y(train==1);
    
    [mu1, ~] = mle(xtrain(ytrain==1, :));
    [mu2, ~] = mle(xtrain(ytrain==-1, :));
    
    xtest = x(test==1, :);
    ytest = y(test==1);
    for i=1:size(xtest, 1)
        tmpXtest = transpose(xtest(i,:));
        tmpYtest = ytest(i);
        
        [~, ~, ytest1] = sph_bayes(tmpXtest, xtrain, ytrain);
        ytest2 = new_classifier(tmpXtest, mu1, mu2);
        
        err1(tmpFold, i) = tmpYtest ~= ytest1;
        err2(tmpFold, i) = tmpYtest ~= ytest2;
    end
end
e1 = mean(mean(err1, 2))
e2 = mean(mean(err2, 2))