function [] = plotIt(X, Y, Y_model, weights, bias, supportVectors)
X_misclassified = X((Y ~= Y_model), :);
Y_misclassified = Y((Y ~= Y_model), :);
hold on
gscatter(X(:,1), X(:, 2), Y, 'br', 'oo'); % correct
plot(X_misclassified(:, 1), X_misclassified(:, 2), 'cx', 'MarkerSize', 10) % misclassified
ezplot(@(x1, x2) weights(1)*x1 + weights(2)*x2 + bias) % hyperplane
plot(supportVectors(:,1),supportVectors(:,2),'ko','MarkerSize',10);% support vectors 
hold off
legend('+1 correct', '-1 correct', 'Misclassified', 'Hyperplane', 'Support Vectors')
title('')