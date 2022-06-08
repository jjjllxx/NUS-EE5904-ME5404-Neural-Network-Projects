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
reg=0:0.05:0.95;
atrain=zeros(1,size(reg,2));
aval=zeros(1,size(reg,2));
pos=1;
% specify the structure and learning algorithm for MLP
for r=reg
    net = patternnet(150,'traingdx');
    net = configure(net,x,y);
    net.performParam.regularization = r; % regularization strength
    net.trainparam.lr=0.01;
    net.trainparam.epochs=10000;
    net.trainparam.goal=1e-6;
    net.divideParam.trainRatio=0.7;
    net.divideParam.valRatio=0.15;
    net.divideParam.testRatio=0.15;
    net.trainParam.max_fail = 50;
    % Train the MLP
    [net,tr]=train(net,x,y);
    
    % accuracy
    pred_train = net(xtrain);
    accu_train = 1 - mean(abs(pred_train-ytrain));
    pred_val = net(xtest);
    accu_val = 1 - mean(abs(pred_val-ytest));
    atrain(pos)=accu_train;
    aval(pos)=accu_val;
    pos=pos+1;
    display(['accu_train:',num2str(accu_train*100)])
    display(['accu_val:',num2str(accu_val*100)])
end
% % show the performance with epoch
% plotperform(tr)
figure
plot(reg, atrain);
hold on
plot(reg, aval);
legend('train:','val:')
title('accuracy under different regularization parameter')