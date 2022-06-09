clear; 
clc; 
close all;
% train data(with noise) and test data
xtrain = -1.6 : 0.08 : 1.6;
N_train = size(xtrain, 2);
ytrain = 1.2 * sin(pi * xtrain)-cos(2.4 * pi * xtrain);
noise = randn([1, N_train]);
ytrain = ytrain + 0.3 * noise;
xtest = -1.6 : 0.01 : 1.6;
N_test = size(xtest, 2);

% calculate phi for train data
phi_train = zeros(N_train, N_train);
sigma = 0.1;
for i = 1 : N_train
    for j = 1 : N_train
        phi_train(i, j) = exp(-0.5 * ((xtrain(i) - xtrain(j)) / sigma) ^ 2);
    end
end
lambda_box=[0, 0.2, 0.5, 1, 2, 5];
error = [];
I = eye(N_train);
for lambda = lambda_box
    weights = (phi_train * phi_train' + lambda * I) \ (phi_train * ytrain');
    weights = weights';
    
    phi_test = zeros(N_test, N_train);
    for i = 1 : N_test
        for j = 1 : N_train
            phi_test(i, j) = exp(-0.5 * ((xtest(i) - xtrain(j)) / sigma) ^ 2);
        end
    end
    ypred = weights * phi_test';
    ytest = 1.2 * sin(pi * xtest)-cos(2.4 * pi * xtest);
    
    % evaluate performance
    error = [error, immse(ytest, ypred)];
    
    % draw train, test, predicted test
    figure
    hold on
    plot(xtrain, ytrain, 'x', 'markersize', 10)
    plot(xtest, ytest, 'linewidth', 2)
    plot(xtest, ypred, 'b', 'linewidth', 2)
    legend('train', 'test', 'predict')
    title(['RBFN with regularization(lambda = ', num2str(lambda),')'])
    hold off
    saveas(gcf, ['RBFN with regularization(lambda = ', num2str(lambda),').png'])
end


