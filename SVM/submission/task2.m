clc; clear; close all;
load('train.mat')
load('test.mat')

% preprocessing the data
train_mean = mean(train_data, 2);
train_s = std(train_data, 0, 2);
x_train = (train_data - train_mean) ./ train_s;
x_test = (test_data - train_mean) ./ train_s;

% load the discriminant function
load('omega.mat')
load('bias.mat')

% hard margin with linear kernel
tr_hard = getAcc(x_train, train_label, omega_hard, bias_hard, 0, x_train, train_label);
te_hard = getAcc(x_test, test_label, omega_hard, bias_hard, 0, x_train, train_label);

% hard margin with polynomial kernel
tr_hardp = zeros(1, 4);
te_hardp = zeros(1, 4);
for i = 1 : 4
    tr_hardp(i) = getAcc(x_train, train_label, omega_hardp(:, i), bias_hardp(i), i + 1, x_train, train_label);  
    te_hardp(i) = getAcc(x_test, test_label, omega_hardp(:, i), bias_hardp(i), i + 1, x_train, train_label);
end

% soft margin with polynomial kernel
tr_soft = zeros(5, 4);
te_soft = zeros(5, 4);
for i = 1 : 5
    for j = 1 : 4
        tr_soft(i, j) = getAcc(x_train, train_label, omega_soft(:, i, j), bias_soft(i, j), i, x_train, train_label);
        te_soft(i, j) = getAcc(x_test, test_label,omega_soft(:, i, j), bias_soft(i, j), i, x_train, train_label);
    end
end

% function to get accuracy
function [acc] = getAcc(data, label, omega, bias, p, train_d, train_l)
if p == 0
    pred = sign(data' * omega + bias);
    acc = sum(pred == label) / size(data, 2);
else
    gx = omega .* train_l .* (train_d' * data + 1) .^ p; 
    gx = sum(gx,1)'; 
    pred = sign(gx + bias);
    acc = sum(pred == label) / size(data, 2);
end
end
