load('medium_100_10k');

REP = 10;
f = zeros(REP, 1);
cavWordIdx = find(strcmp(vocab, 'cavalry'));
for rep=1:REP
    IDX = kmeans(wordembeddings, 10, 'Replicates', 1);
    cavCluster = IDX(cavWordIdx);
    wordsInCluster = IDX==cavCluster;
    
    wordCount = sum(wordsInCluster);
    N0 = nchoosek(wordCount, 2);
    
    IDX = kmeans(wordembeddings, 10, 'Replicates', 1);
    wordsInCluster2 = IDX==cavCluster;
    
    sameWords = and(wordsInCluster, wordsInCluster2);
    wordCount = sum(sameWords);
    if (wordCount >= 2)
        N1 = nchoosek(wordCount, 2);
    else
        N1 = 0; % If there are less than two same words there are no pairs
    end
    
    f(rep) = N1 / N0;
end
sprintf('The average fraction of words pairs remaining in the same cluster is %f.', mean(f))