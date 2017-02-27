load('d2.mat')

%% a
C = 1
svmStruct = svmtrain(X,Y, 'BoxConstraint', C, 'autoscale', false);
classify = svmclassify(svmStruct,X);

w = svmStruct.Alpha' * svmStruct.SupportVectors;
figure()
plotIt(X, Y, classify, w, svmStruct.Bias, svmStruct.SupportVectors);

% Fixed: X, Y
% Parameters: C, Kernel (linear, quadratic, rbf), OptMethod(SMO, QP)
% Output: execution time, classification accuracy (%)

%% SMO
% 1. Linear Kernel
[SMO_L_Time, SMO_L_Accuracy] = svm_crossval(5, X, Y, C, 'linear', 'SMO', false, false)
% 2. Quadratic Kernel
[SMO_Q_Time, SMO_Q_Accuracy] = svm_crossval(5, X, Y, C, 'quadratic', 'SMO', false, false)    
% 3. RBF Kernel
[SMO_RBF_Time, SMO_RBF_Accuracy] = svm_crossval(5, X, Y, C, 'rbf', 'SMO', false, false)
%% QP
% 1. Linear Kernel
[QP_L_Time, QP_L_Accuracy] = svm_crossval(5, X, Y, C, 'linear', 'QP', false, false)
% 2. Quadratic Kernel
[QP_Q_Time, QP_Q_Accuracy] = svm_crossval(5, X, Y, C, 'quadratic', 'QP', false, false)    
% 3. RBF Kernel
figure()
[QP_RBF_Time, QP_RBF_Accuracy] = svm_crossval(5, X, Y, C, 'rbf', 'QP', true, false)