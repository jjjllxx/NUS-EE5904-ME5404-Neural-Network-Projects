clear all;
clc;
load('xtest.mat');
load('xtrain.mat');
ytrain=[ones(1,450),zeros(1,450)];
ytest=[ones(1,50),zeros(1,50)];
xtrain = xtrain';
xtest = xtest';

x=[xtrain, xtest];
y=[ytrain, ytest];

% specify the structure and learning algorithm for MLP
net = patternnet(150,'traingdx');
net = configure(net,x,y);

net.trainparam.lr=0.01;
net.trainparam.epochs=10000;
net.trainparam.goal=1e-6;
net.divideParam.trainRatio=1.0;
net.divideParam.valRatio=0.0;
net.divideParam.testRatio=0;

% Train the MLP
[net,tr]=train(net,x,y);

% accuracy
pred_train = net(xtrain);
accu_train = 1 - mean(abs(pred_train-ytrain));
pred_val = net(xtest);
accu_val = 1 - mean(abs(pred_val-ytest));
display(['accu_train:',num2str(accu_train*100)])
display(['accu_val:',num2str(accu_val*100)])
% show the performance with epoch
plotperform(tr)
