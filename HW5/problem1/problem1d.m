load('hw5_p1b.mat');

[~, z_lin]  = linearKmeans(X, 2, inf);

sigma=0.2;
kernel = @(x1, x2) exp(-norm(x1 - x2)^2 / (2 * sigma^2)); % gaussian RBF kernel
z_rbf = kernelKmeans(X, kernel, 2, inf);

fig_lin = figure();
gscatter(X(:,1), X(:, 2), z_lin(:, 1));


fig_rbf = figure();
gscatter(X(:, 1), X(:, 2), z_rbf(:, 1));

saveas(fig_lin, 'problem1d_lin.jpg')
saveas(fig_rbf, 'problem1d_rbf.jpg')