load('dataset1.mat');


fig = figure;
hold on
title('Scatterplot for dataset X')
len = length(x);
[mu, sigma] = sge(x);
scatter(x(:,1),x(:,2),[10],'filled');
c = [];
for k = 1:1:3
    c = [c circle(mu(1,1),mu(1,2),k*sigma)];      
end
[k1,k2,k3] = distance(x,mu,1*sigma,2*sigma,3*sigma);
legend(c,sprintf('k: 1, %.4f',1-k1/len),sprintf('k: 2, %.4f',1-k2/len),sprintf('k: 3, %.4f',1-k3/len));

saveas(fig,'./hw1_2_1.png')