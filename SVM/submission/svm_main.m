load('train.mat')
load('test.mat')

% preprocessing the data
train_mean = mean(train_data, 2);
train_s = std(train_data, 0, 2);
x_train = (train_data - train_mean) ./ train_s;
x_test = (test_data - train_mean) ./ train_s;
x_eval = (eval_data - train_mean) ./ train_s;
N_train = size(x_train, 2);
N_test = size(x_test, 2);
N_eval = size(x_eval, 2);

% hyperparameter of Gaussian kernel
sigma = 30;
gamma = 0.5 / (sigma ^ 2);
C = 400;
% judge the mercer condition
xx=zeros(N_train, N_train);
for i=1 : N_train
    for j=1 : N_train
        xx(i, j) = exp(-gamma * sum((x_train(:,i) - x_train(:,j)) .^ 2));
    end
end
eig_v = eig(xx);
if min(eig_v) < -1e-4
    fprintf(['Gaussian kernel(sigma = ', num2str(sigma),' C = ', num2str(C),') is not admissible \n'])
else
    fprintf(['Gaussian kernel(sigma = ', num2str(sigma),' C = ', num2str(C),') is admissible \n'])
end

% calculate the H matrix
dd = train_label * train_label';
H_p = dd .* xx;
[alpha_main, bias_main] = getDisGau(H_p, C, x_train, train_label, gamma);

[~, tr_acc] = getAccPred(x_train, train_label, alpha_main, bias_main, gamma, x_train, train_label);
[~, te_acc] = getAccPred(x_test, test_label, alpha_main, bias_main, gamma, x_train, train_label);
% eval_predicted
[eval_predicted, ev_acc] = getAccPred(x_eval, eval_label,alpha_main, bias_main, gamma, x_train, train_label);

% function to get discriminant function
function [omega, bias] = getDisGau(h_matrix, C, x_train, train_label, gamma)
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
a_idx = find(alpha > 1e-8);

a_num = size(a_idx, 1);
selected = train_label(a_idx);
bias_all = zeros(a_num, 1);

for i=1:a_num
    gx = alpha .* train_label .* exp(-gamma * sum((x_train-x_train(:, a_idx(i))) .^ 2, 1)');
    bias_all(i) = selected(i) - sum(gx);
end
omega = alpha;
bias = sum(bias_all) / a_num;
end

% function to get accuracy
function [pred, acc] = getAccPred(data, label, alpha, bias, gamma, train_d, train_l)
N = size(data, 2);
pred = zeros(N, 1);
for i = 1 : N
    gx = alpha .* train_l .* exp(-gamma * sum((train_d - data(:,i)) .^ 2, 1)');
    pred(i) = sum(gx) + bias;
end
pred = sign(pred);
acc = sum(pred == label) / size(data, 2);
end
