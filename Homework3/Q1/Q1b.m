clear; 
clc; 
close all;
xtrain = -1.6 : 0.08 : 1.6;
N_train = size(xtrain, 2);
ytrain = 1.2 * sin(pi * xtrain)-cos(2.4 * pi * xtrain);
noise = randn([1, N_train]);
ytrain = ytrain + 0.3 * noise;
xtest = -1.6 : 0.01 : 1.6;
N_test = size(xtest, 2);

N_sample = 20;
p = randperm(N_train);
chosen = p(1:N_sample);
dmax = 0;
for i = 1 : N_sample
    for j = i + 1 : N_sample
        dmax = max(dmax, abs(xtrain(chosen(i)) - xtrain(chosen(j))));
    end
end
dmax = dmax * dmax;

phi_train = zeros(N_sample, N_train);
for i = 1 : N_train
    for j = 1 : N_sample
        phi_train(j, i) = exp(-1 * N_sample / dmax * (xtrain(chosen(j)) - xtrain(i))^2);
    end
end
weights = ytrain / phi_train;

phi_test = zeros(N_sample, N_test);

for i = 1 : N_test
    for j = 1 : N_sample
        phi_test(j, i) = exp(-1 * N_sample / dmax * (xtrain(chosen(j)) - xtest(i))^2);
    end
end
ypred = weights * phi_test;
ytest = 1.2 * sin(pi * xtest)-cos(2.4 * pi * xtest);

% evaluate performance
error = immse(ytest, ypred);

% draw train, test, predicted test
figure
hold on
plot(xtrain, ytrain, 'x', 'markersize', 10)
plot(xtest, ytest, 'linewidth', 2)
plot(xtest, ypred, 'b', 'linewidth', 2)
legend('train', 'test', 'predict')
title('RBFN(Fixed Centers Selected at Random)')
hold off
saveas(gcf, 'RBFN(Fixed Centers Selected at Random).png')
