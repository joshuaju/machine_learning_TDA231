function[time, accuracy] = svm_crossval(nfolds, X, Y, C, kernel, method, plotTrain, plotClassify);
% C is box constrain
% kernel is kernel function: linear, quadratic or rbf
% method: SMO or QP
indicies = crossvalind('Kfold', size(Y, 1), nfolds);
accuracy = zeros(nfolds, 1);
tic
for tmpFold=1:nfolds
    test = (indicies == tmpFold); train = ~test;
    
    Xtrain = X(train==1, :);
    Ytrain = Y(train==1);
    Xtest = X(test==1, :);
    Ytest = Y(test==1, :);
    
    svmStruct = svmtrain(Xtrain, Ytrain, 'boxconstraint', C, 'kernel_function', kernel, 'method', method, 'showplot', plotTrain);
    classified = svmclassify(svmStruct,Xtest, 'showplot', plotClassify);
    accuracy(tmpFold) = sum(Ytest== classified) / size(Ytest, 1);
end
time = toc;
accuracy = mean(accuracy);
end