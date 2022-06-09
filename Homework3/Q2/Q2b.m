clear; clc; close all; 
load mnist_m.mat;
% matric number A0232696W
% process trainset
TrN = size(train_data, 2);
TrLabel = zeros(1,TrN); 
TrLabel(train_classlabel == 6 | train_classlabel == 9) = 1;

% process testset
myTestx = test_data;
TeN = size(myTestx, 2);
TeLabel = zeros(1, TeN);
TeLabel(test_classlabel == 6 | test_classlabel == 9) = 1;

% choose randomly
N_sample = 100;
p = randperm(TrN);
chosen = p(1 : N_sample);
myTrainx = train_data;
TrN = size(myTrainx, 2);

width_box = [0, 0.1, 1, 10, 100, 1000, 10000];
for dmax = width_box
    if dmax == 0
        for i = 1 : N_sample
            for j = i + 1 : N_sample
                dmax = max(dmax, sum((myTrainx(:, chosen(i)) - myTrainx(:, chosen(j))) .^ 2));
            end
        end
        dmax = dmax * dmax;
    end
    
    phi_train = zeros(TrN, N_sample);
    for i = 1 : TrN
        for j = 1 : N_sample
            phi_train(i, j) = exp(-N_sample / dmax * sum((myTrainx(:, i) - myTrainx(:, chosen(j))) .^ 2));
        end
    end
    weights = TrLabel / phi_train';
    TrPred = weights * phi_train';
    
    phi_test = zeros(TeN, N_sample);
    for i = 1 : TeN
        for j = 1 : N_sample
            phi_test(i, j) = exp(-N_sample / dmax * sum((myTestx(:, i) - myTrainx(:, chosen(j))) .^ 2));
        end
    end
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
    if dmax <= 10000
        title(['Fixed Centers Selected at Random(width =', num2str(dmax), ')'])
        % save the figure
        saveas(gcf, ['Fixed Centers Selected at Random(width =', num2str(dmax), ').png'])
    else
        title(['Fixed Centers Selected at Random(width =', num2str(dmax), ')'])
        % save the figure
        saveas(gcf, 'Fixed Centers Selected at Random(appropriate size).png')
    end
end
