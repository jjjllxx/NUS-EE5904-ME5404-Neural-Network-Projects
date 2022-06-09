clear; clc; close all; 
load mnist_m.mat;
% matric number A0232696W
% process trainset
myTrainx = train_data;
TrN = size(myTrainx, 2);
TrLabel = zeros(1,TrN); 
TrLabel(train_classlabel == 6 | train_classlabel == 9) = 1;

% process testset
myTestx = test_data;
TeN = size(myTestx, 2);
TeLabel = zeros(1, TeN);
TeLabel(test_classlabel == 6 | test_classlabel == 9) = 1;

% initialize the centre randomly
center_num = 2;
p = randperm(TrN);
chosen = p(1 : center_num);
center1 = myTrainx(:,chosen(1))';
center2 = myTrainx(:,chosen(2))';
iter = 0;
while iter < 20
    pos1 = 1;
    pos2 = 1;
    cluster1 = zeros(1, 784);
    cluster2 = zeros(1, 784);
    idx1 = zeros(1);
    idx2 = zeros(1);
    for i = 1 : TrN
        d1 = sum((myTrainx(:,i)' - center1) .^ 2);
        d2 = sum((myTrainx(:,i)' - center2) .^ 2);
        % compare each sample's distance with two center
        if d1 < d2
            cluster1(pos1, :) = myTrainx(:, i)';
            idx1(pos1) = i;
            pos1 = pos1 + 1;
            
        else
            cluster2(pos2, :) = myTrainx(:, i)';
            idx2(pos2) = i;
            pos2 = pos2 + 1;
        end
    end
    % update 2 centers
    center1 = mean(cluster1);
    center2 = mean(cluster2);
    iter = iter + 1;
end

% calculate RBFN weights
phi_train = zeros(center_num, TrN);
width = -center_num / sum((center1 - center2) .^ 2);
for i = 1 : TrN
    phi_train(1, i) = exp(width * sum((myTrainx(:, i) - center1') .^ 2));
    phi_train(2, i) = exp(width * sum((myTrainx(:, i) - center2') .^ 2));
end
weights = TrLabel / phi_train;
phi_test = zeros(TeN, center_num);

for i = 1 : TeN 
    phi_test(i, 1) = exp(width * sum((myTestx(:, i) - center1') .^ 2));
    phi_test(i, 2) = exp(width * sum((myTestx(:, i) - center2') .^ 2));
end
TrPred = weights * phi_train;
TePred = weights * phi_test';

%evaluate performance
TrAcc = zeros(1,1000);
TeAcc = zeros(1,1000);
thr = zeros(1,1000);
for i = 1:1000
    t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
    thr(i) = t;
    TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
    TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
end
figure
plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
title('RBFN performance')
saveas(gcf, 'RBFN performance.png')

% draw result
enlarge_size = 8;
final_center1 = imresize(reshape(center1, 28, 28), enlarge_size);
final_center2 = imresize(reshape(center2, 28, 28), enlarge_size);

figure
imshow([final_center1, final_center2]);
title('Clustering result')
mean_val1 = mean(myTrainx(:, TrLabel == 1), 2);
mean_val2 = mean(myTrainx(:, TrLabel == 0), 2);
saveas(gcf, 'Clustering result.png')

mean_val1 = imresize(reshape(mean_val1', 28, 28), enlarge_size);
mean_val2 = imresize(reshape(mean_val2', 28, 28), enlarge_size);
figure
imshow([mean_val1, mean_val2]);
title('Mean of training image')
saveas(gcf, 'Mean of training image.png')
