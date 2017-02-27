load('d1b.mat');
%% a Train
C = 1
svmStruct = svmtrain(X,Y, 'BoxConstraint', C, 'autoscale', false);

%% b
Y_model = svmclassify(svmStruct,X);


w = svmStruct.Alpha' * svmStruct.SupportVectors

X_correct = X((Y == Y_model),:)
Y_correct = Y((Y == Y_model), :);
X_misclassified = X((Y ~= Y_model), :);
Y_misclassified = Y((Y ~= Y_model), :);

hold on
gscatter(X_correct(:,1), X_correct(:, 2), Y_correct, 'br', 'oo'); % correct
gscatter(X_misclassified(:, 1), X_misclassified(:, 2), Y_misclassified, 'rb', 'xx') % not correct
ezplot(@(x1, x2) w(1)*x1 + w(2)*x2 + svmStruct.Bias) % hyperplane
plot(svmStruct.SupportVectors(:,1),svmStruct.SupportVectors(:,2),'o','MarkerSize',10);% support vectors 
hold off
legend('+1 correct', '-1 correct', '-1 false', 'Hyperplane', 'Support Vectors')
title('Problem 2.1')


%% c
bias = svmStruct.Bias

%% d
%        err_rate = sum(Y(P.test)~= C)/P.TestSize % mis-classification rate
%        conMat = confusionmat(Y(P.test),C) % the confusion matrix
soft_margin = 2 / norm(w)