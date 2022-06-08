clear all;clc;
load('xtest.mat');
load('xtrain.mat');
ytrain=[ones(1,450),zeros(1,450)];
ytest=[ones(1,50),zeros(1,50)];
xtrain = xtrain';
xtest = xtest';
image=[xtrain,xtest];
label=[ytrain,ytest];
train_num=size(xtrain,2);
epoch=50;
[net, accu_train, accu_val]=train_seq(150, image, label, train_num, epoch);

function [net, accu_train, accu_val] = train_seq(n,images,labels,train_num,epochs)
% Construct a 1-n-1 MLP and conduct sequential training.

% Args:
% n: int, number of neurons in the hidden layer of MLP.
% images: matrix of (image_dim, image_num), containing possibly preprocessed image data as input.
% labels: vector of (1, image_num), containing corresponding label of each image.
% train_num: int, number of training images.
% val_num: int, number of validation images.
% epochs: int, number of training epochs.

% Returns:
% net: object, containing trained network.
% accu_train: vector of (epochs, 1), containing the accuracy on training
% set of each epoch during trainig.
% accu_val: vector of (epochs, 1), containing the accuracy on validation
% set of each eopch during trainig.

% 1. Change the input to cell array form for sequential training
images_c = num2cell(images, 1);
labels_c = num2cell(labels, 1);

% 2. Construct and configure the MLP
net = patternnet(n);
net.divideFcn = 'dividetrain'; % input for training only
net.performParam.regularization = 0.25; % regularization strength
net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
net.trainParam.epochs = epochs;
accu_train = zeros(epochs,1); % record accuracy on training set of each epoch
accu_val = zeros(epochs,1); % record accuracy on validation set of each epoch

% 3. Train the network in sequential mode
for i = 1 : epochs
    display(['Epoch: ', num2str(i)])
    idx = randperm(train_num); % shuffle the input
    net = adapt(net, images_c(:,idx), labels_c(:,idx));
    pred_train = round(net(images(:,1:train_num))); % predictions on training set
    accu_train(i) = 1 - mean(abs(pred_train-labels(1:train_num)));
    pred_val = round(net(images(:,train_num+1:end))); % predictions on validation set
    accu_val(i) = 1 - mean(abs(pred_val-labels(train_num+1:end)));
end
figure
plot(1:epochs, accu_train)
title('accuracy of training(sequential mode)')
figure
plot(1:epochs, accu_val)
title('accuracy of validation(sequential mode)')
end
