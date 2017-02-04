load('digits.mat');

%y = reshape(data(:,45,3),16,16);
%imshow(y);
d5 = data(:,:,5);
d8 = data(:,:,8);
mu5 = mean(d5);
mu8 = mean(d8);

[Ytest] = new_classifier([d5;d8], mu5, mu8)