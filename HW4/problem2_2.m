load('d2.mat')

%% a
svmStruct = svmtrain(X,Y,'showplot',true);
C = svmclassify(svmStruct,X(P.test,:),'showplot',true);
%        err_rate = sum(Y(P.test)~= C)/P.TestSize % mis-classification rate
%        conMat = confusionmat(Y(P.test),C) % the confusion matrix
