clear; clc; close all;
load('Digits.mat')
% my matric number A0232696W omit class 1 and class 2
% process train set
trIdx = find(train_classlabel == 0 | train_classlabel == 3 | train_classlabel == 4);
myTrain = train_data(:, trIdx);
TrLabel = train_classlabel(trIdx);
% process test set
teIdx = find(test_classlabel == 0 | test_classlabel == 3 | test_classlabel == 4);
myTest = test_data(:, teIdx);
TeLabel = test_classlabel(teIdx);
TrN = size(myTrain, 2);
TeN = size(myTest, 2);
% SOM parameter
iter = 0; N = 1500;
record = [0, 10, 20, 50, 100:100:N]; r = 1;
N_weights = 100;
weights = rand(784, N_weights);
sigma0 = N_weights / 2;
tau = N / log(sigma0);
TeAcc = zeros(1, size(record, 2)); TrAcc = zeros(1, size(record, 2));
while iter <= N
    chosen = randi(TrN);
    now = myTrain(:, chosen);
    % find the index of minimum
    [~, idx] = min(sum((now - weights) .^ 2));
    sigma = 2 * (sigma0 * exp(-iter / tau)) ^ 2 ;
    learning_rate = 0.1 * exp(iter / N);
    for i = 1 : N_weights
        d = (fix((i - 1) / 10) - fix((idx - 1) / 10)) ^ 2 + (mod(i - 1, 10) - mod(idx - 1, 10)) ^ 2;
        h = exp(-d / sigma);
        weights(:, i) = weights(:, i) + learning_rate * h * (now - weights(:, i));
    end
    if iter == record(r)
        vote = zeros(5, N_weights);
        for i = 1 : TrN
            now = myTrain(:, i);
            [~, idx] = min(sum((now - weights) .^ 2));
            vote(TrLabel(i) + 1, idx) = vote(TrLabel(i) + 1, idx) + 1;
        end
        % calculate each neuron's label
        neurons_label = zeros(1, N_weights);
        neurons_val = zeros(1, N_weights);
        for i = 1 : N_weights
            [val, idx] = max(vote(:, i));
            neurons_label(i) = idx - 1;
            neurons_val(i) = val;
        end
        % calculate test accuracy
        for i = 1 : TeN
            now = myTest(:, i);
            [~, idx] = min(sum((now - weights) .^ 2));
            TeAcc(r) = TeAcc(r) + (neurons_label(idx) == TeLabel(i));
        end
        for i = 1 : TrN
            now = myTrain(:, i);
            [~, idx] = min(sum((now - weights) .^ 2));
            TrAcc(r) = TrAcc(r) + (neurons_label(idx) == TrLabel(i));
        end
        TeAcc(r) = TeAcc(r) / TeN;
        TrAcc(r) = TrAcc(r) / TrN;
        r = r + 1;
    end
    iter = iter + 1;
end

% visualize weights
trained_weights = [];
for i = 0 : 9
    weights_row = [];
    for j = 1 : 10
        weights_row = [weights_row, reshape(weights(:, i*10+j), 28, 28)];
    end
    trained_weights = [trained_weights; weights_row];
end
figure
imshow(imresize(trained_weights, 4))
title('weights visualization')
saveas(gcf, 'weights visualization.png')

% show the conceptual map
neurons_label = reshape(neurons_label, 10, 10)';
neurons_val = neurons_val/max(neurons_val);
neurons_val = reshape(neurons_val, [10,10])';
figure
img = imagesc(neurons_label);
img.AlphaData = neurons_val;
for i = [0, 3, 4]
    neurons_label(neurons_label == i) = num2str(i);
end
label = num2str(neurons_label, '%s');       
[x, y] = meshgrid(1:10);  
hStrings = text(x(:), y(:), label(:), 'HorizontalAlignment', 'center');
title('conceptual map')
saveas(gcf, 'conceptual map.png')

% show the accuracy
figure
hold on
plot(record, TrAcc, 'linewidth', 2)
plot(record, TeAcc, 'linewidth', 2)
hold off
legend('Train Accuracy','Test Accuracy', 'Location', 'Best')
title('accuracy according to iteration')
saveas(gcf, 'accuracy according to iteration.png')