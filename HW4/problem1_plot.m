data_pos = [2 2; 4 4; 4 0];
data_neg = [0 0; 2 0; 0 2];

hold on
axis([-1, 5, -1, 5]);
scatter(data_pos(:, 1), data_pos(:, 2), 'b', 'filled');
scatter(data_neg(:, 1), data_neg(:, 2), 'r', 'filled');
plot([-1;4], [4;-1], 'k-');
plot([-1;3], [3;-1], 'r--');
plot([-1;5], [5;-1], 'b--');