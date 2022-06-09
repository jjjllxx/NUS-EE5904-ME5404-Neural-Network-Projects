clear; clc; close all;
x = linspace(-pi,pi,400);
trainX = [x; sinc(x)]; 
N_train = size(trainX, 2);
iter = 0;N = 500;
record = [0, 10, 20, 50, 100:100:N]; r = 1;
N_weights = 40;
weights = rand(2, N_weights);
sigma0 = N_weights / 2;
tau = N / log(sigma0);
while iter <= N
    chosen = randi(N_train);
    now = trainX(:, chosen);
    [~, idx] = min(sum((now - weights) .^ 2));
    sigma = 2 * (sigma0 * exp(-iter / tau)) ^ 2;
    learning_rate = 0.1 * exp(iter / N);
    for i = 1 : N_weights
        d = i - idx;
        h = exp(-d^2 / sigma);
        weights(:, i) = weights(:, i) + learning_rate * h * (now - weights(:, i));
    end
    % draw and save figure of certain iterations
    if iter == record(r)
        figure
        hold on
        plot(weights(1,:), weights(2,:), '+b-');
        plot(trainX(1,:), trainX(2,:), '+r');
        title(['1-dim SOM result(iteration=', num2str(iter), ')'])
        hold off
        axis equal
        saveas(gcf, ['1-dim SOM result(iteration=', num2str(iter), ').png'])
        r = r + 1;
    end
    iter = iter + 1;
end
