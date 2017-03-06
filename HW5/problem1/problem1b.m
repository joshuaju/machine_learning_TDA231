load('hw5_p1a');
[mu_2 z_2] = linearKmeans(X, 2, 2)
[mu_inf z_inf]= linearKmeans(X, 2, inf)

z_diff = z_2(:, 1) ~= z_inf(:, 1);
X_diff = X(z_diff, :);

f = figure(); hold on
gscatter(X(:, 1), X(:, 2), z_inf(:, 1), 'br'); % assigned points
plot(X_diff(:, 1), X_diff(:, 2), 'ko', 'MarkerSize', 10); % assignment changed
legend('Class 1', 'Class 2', 'Assignment changed');

saveas(f, 'problem1b.jpg')
