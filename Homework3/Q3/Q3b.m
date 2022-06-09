clear; clc; close all;
X = randn(800,2);
s2 = sum(X.^2,2);
trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';
N_train = size(trainX, 2);
% SOM parameter
iter = 0;
N = 500;
record = [0, 10, 20, 50, 100:100:N]; r = 1;
N_weights = 8 * 8;
weights = rand(2, N_weights);
sigma0 = N_weights / 2;
tau = N / log(sigma0);
while iter <= N
    chosen = randi(N_train);
    now = trainX(:, chosen);
    % find the index of minimum
    [~, idx] = min(sum((now - weights) .^ 2));
    sigma = 2 *(sigma0 * exp(-iter / tau)) ^ 2 ;
    learning_rate = 0.1 * exp(iter / N);
    for i = 1 : N_weights
        d = (fix((i - 1) / 8) - fix((idx - 1) / 8)) ^ 2 + (mod(i - 1, 8) - mod(idx - 1, 8)) ^ 2;
        h = exp(-d / sigma);
        weights(:, i) = weights(:, i) + learning_rate * h * (now - weights(:, i));
    end
    % draw and save figure of certain iterations
    if iter == record(r)
        figure
        hold on
        plot(trainX(1,:),trainX(2,:),'+r');
        for i = 1 : 8
            plot(weights(1, i*8-7:i*8), weights(2, i*8-7:i*8), '+b-');
            plot(weights(1, i:8:end), weights(2, i:8:end), '+b-');
        end
        hold off
        title(['2-dim SOM result(iteration=', num2str(iter), ')'])
        axis equal
        saveas(gcf, ['2-dim SOM result(iteration=', num2str(iter), ').png'])
        r = r + 1;
    end
    iter = iter + 1;
end
