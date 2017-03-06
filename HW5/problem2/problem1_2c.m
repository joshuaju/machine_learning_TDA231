load('medium_100_10k');

N = 1000;   % sample size
k = 10;     % k-means

samples = datasample(wordembeddings, N, 'Replace', false);
IDX = kmeans(samples, k);
samples2D = tsne(samples);

fig = figure();
gscatter(samples2D(:, 1), samples2D(:, 2), IDX, '', '.ox+*sdv^p');
saveas(fig, 'problem2c.jpg')