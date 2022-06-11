clc; clear; close all;
load('train.mat')
load('test.mat')

% preprocessing the data
train_mean = mean(train_data, 2);
train_s = std(train_data, 0, 2);
x_train = (train_data - train_mean) ./ train_s;
x_test = (test_data - train_mean) ./ train_s;

N_train = size(x_train, 2);
N_test = size(x_test, 2);

% calculate the H matrix
dd = train_label * train_label';
xx = x_train' * x_train;
H_hard = dd .* xx;

xx1 = ones(N_train, N_train);
xx = xx + 1;
% judge the mercer condition
H_p = zeros(5, N_train, N_train);
for p = 1 : 5
    xx1 = xx1 .* xx;
    eig_v = eig(xx1);
    if min(eig_v) < -1e-4
        fprintf(['polynomial kernel(p = ', num2str(p), ') is not admissible \n'])
    end
    H_p(p, :, :) = dd .* xx1;
end

% hard margin with linear kernel
[omega_hard, bias_hard] = getDis(H_hard, 1e6, x_train, train_label, 0);
fprintf(['\n', 'hard linear kernel finish', '\n']);

% hard margin with polynomial kernel
omega_hardp = zeros(2000, 4);
bias_hardp = zeros(1, 4);
for i = 2 : 5
    [omega_hardp(:, i - 1), bias_hardp(i - 1)] = getDis(H_p(i, :, :), 1e6, x_train, train_label, i);
    fprintf(['\n hard polynomial kernel(p=',num2str(i),') finished \n']);
end

% soft margin with polynomial kernel
omega_soft = zeros(2000, 5, 4);
bias_soft = zeros(5, 4);
C_box = [0.1, 0.6, 1.1, 2.1];
for i = 1 : 5
    for j = 1 : 4
        [omega_soft(:,i, j), bias_soft(i, j)] = getDis(H_p(i, :, :), C_box(j), x_train, train_label, i);
        fprintf(['\n soft polynomial kernel(p=',num2str(i), ', C=',num2str(C_box(j)),')finished \n']);
    end
end

% save the paramters of discriminant function
save('omega.mat', 'omega_hard', 'omega_hardp', 'omega_soft')
save('bias.mat', 'bias_hard', 'bias_hardp', 'bias_soft')

% function to get discriminant function
function [omega, bias] = getDis(h_matrix, C, x_train, train_label, p)
N_train = size(x_train, 2);
h_matrix = reshape(h_matrix, [N_train, N_train]);

f = -ones(1, N_train);
A = [];
b = [];

Aeq = train_label';
beq = 0;

lb = zeros(N_train, 1);
ub = ones(N_train, 1) * C;

x0 = [];
opt = optimset('LargeScale', 'off', 'MaxIter', 1000);

alpha = quadprog(h_matrix, f, A, b, Aeq, beq, lb, ub, x0, opt);
a_idx = find(alpha > 1e-4);

if p == 0
    omega = sum(alpha' .* train_label' .* x_train, 2);
    bias = mean(1 ./ train_label(a_idx) - x_train(:,a_idx)' * omega);
else
    omega = alpha;
    gx = alpha .* train_label .* (x_train' * x_train + 1) .^ p;
    gx = sum(gx, 1)';
    bias = mean(train_label(a_idx) - gx(a_idx));
end
end
