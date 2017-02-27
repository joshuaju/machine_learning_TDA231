load('d1b.mat');
%% a Train
C = 1
svmStruct = svmtrain(X,Y, 'BoxConstraint', C, 'autoscale', false);

%% b
Y_model = svmclassify(svmStruct,X);


w = svmStruct.Alpha' * svmStruct.SupportVectors

plotIt(X, Y, Y_model, w, svmStruct.Bias, svmStruct.SupportVectors);

%% c
bias = svmStruct.Bias

%% d
%        err_rate = sum(Y(P.test)~= C)/P.TestSize % mis-classification rate
%        conMat = confusionmat(Y(P.test),C) % the confusion matrix
soft_margin = 2 / norm(w)