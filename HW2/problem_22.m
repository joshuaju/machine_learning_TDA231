load('digits.mat');

c5samples = data(:,:,5);
c8samples = data(:,:,8);

x = [c5samples c8samples];
y = [repmat(1, 1100, 1); repmat(-1, 1100, 1)]; % 1 correspond to class 5, -1 to class 8

nfolds = 5;
nObservations = length(y);
indicies = crossvalind('Kfold', nObservations, nfolds);

% error matrices: rows correspond to fold, columns to test points
err = zeros(nfolds, 1);
scaledErr = zeros(nfolds, 1);
for tmpFold=1:nfolds
    test = (indicies == tmpFold); train = ~test;
    
    xtrain = x(:, train == 1);
    ytrain = y(train == 1);
   
    mu5 = mean(xtrain(:, ytrain==1), 2);
    mu8 = mean(xtrain(:, ytrain==-1), 2);
    
    scaledMu5 = getMeanFeatureVector(xtrain(:, ytrain==1));
    scaledMu8 = getMeanFeatureVector(xtrain(:, ytrain==-1));
   
    xtest = x(:, test == 1);
    ytest = y(test == 1);     
    for i=1:size(xtest,2)
        tmpXtest = xtest(:, i);        
        scaledtmpXtest = getFeatureVector(tmpXtest);
        
        tmpY = new_classifier(tmpXtest, mu5, mu8);
        scaledtmpY = new_classifier(scaledtmpXtest, scaledMu5, scaledMu8);
        
        
        err(tmpFold, i) = tmpY ~= ytest(i);
        scaledErr(tmpFold, i) = scaledtmpY ~= ytest(i);        
    end    
end
e1 = mean(mean(err, 2))
e2 = mean(mean(scaledErr, 2))


function meanFeatureVector = getMeanFeatureVector(x)
% calculate row-mean for all feature vectors
    vecs = arrayfun(@(i) getFeatureVector(x(:, i)), 1:size(x,2), 'UniformOutput', false);
    vecs = cell2mat(vecs);
    meanFeatureVector = mean(vecs, 2);
end

function featureVector = getFeatureVector(img)
% featureVector: [var(row1), ..., var(rowN), var(col1), ..., var(colN)]
    n = 16;
    m = reshape(img, n, n) / 255;  
    featureVector = zeros(2*n, 1);
    for i=1:n
        featureVector(i) = var(m(i, :)); % var of row
        featureVector(i+n) = var(m(:, i)); % var of column
    end
end