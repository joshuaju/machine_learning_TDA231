load('d1b.mat');

%% 2.1

% a) train model
model = fitcsvm(X, Y, 'KernelFunction', 'linear');
boxConstraint = unique(model.BoxConstraints)

% b) plotting
sv = model.SupportVectors;
sv_idx = find(ismember(X,sv,'rows') == 1);


w = sum(model.Alpha .* Y(sv_idx) .* X(sv_idx, :), 1);        % weights
hpFun = @(x1, x2) w(1)*x1 + w(2)*x2 + model.Bias;           % hyperplane function

eval = transpose(arrayfun(@(i) hpFun(X(i,1), X(i, 2)), 1:size(X, 1)));
mc_points = find((Y .* eval) < 0); % all negative values are misclassified


hold on
H = ezplot(hpFun, [-3.5 4 -2 3]); H.LineColor = 'black';    % hyperplane
gscatter(X(:,1),X(:,2),Y, 'br','ox');                       % data points
plot(X(mc_points,1),X(mc_points,2),'ms','MarkerSize',15);   % misclassified points
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);                 % support vectors

legend('Hyperplane', '+1','-1', 'Misclassified', 'Support Vector(s)');
title('Problem 2.1');
hold off

%% c
bias = model.Bias

%% d
% calculate greatest distance from hyperplane to all misclassified points
num = abs(eval(mc_points));
denom = sqrt(w*w');
distance = num ./ repmat(denom, size(num, 1), 1);

soft_margin = 2 * max(distance)


