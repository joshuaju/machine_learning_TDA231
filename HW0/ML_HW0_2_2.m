%load dataset into
X = load('dataset0.txt');

%normalize X for each column between [0,1]
Y = X - min(X);
Y = Y ./ max(Y);

%covariance of X and Y
covX = cov(X);
covY = cov(Y);
%correlation of x and Y
corrX = corrcoef(X);
corrY = corrcoef(Y);

%plot the colormaps
figure;
colormap default;

subplot(2,2,1)
imagesc(covX);
title('Covariance X')
colorbar

subplot(2,2,2)
imagesc(covY);
title('Covariance Y')
colorbar

subplot(2,2,3)
imagesc(corrX);
title('Correlation  X')
colorbar

subplot(2,2,4)
imagesc(corrY);
title('Correlation  Y')
colorbar

figure;
minCorrY = min(corrY(:));
[minCorrRow,minCorrCol] = find(corrY == minCorrY);
scatter(Y(:,minCorrRow(1)),Y(:,minCorrCol(1)));
title(sprintf('Feature indices %i and %i with correlation value %0.3f',minCorrRow(1),minCorrCol(1),minCorrY));

