load('medium_100_10k');

[~, ~, ~, D] = kmeans(wordembeddings, 10);
[~, I] = min(D);
disp('Words with closest distance (to centroid 1-10) are: ')
disp(vocab(I))