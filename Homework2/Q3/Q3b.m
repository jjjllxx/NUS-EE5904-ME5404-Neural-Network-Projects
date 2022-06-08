clear all;clc;
load('xtest.mat');
load('xtrain.mat');
ytrain=[ones(450,1);zeros(450,1)];
ytest=[ones(50,1);zeros(50,1)];
xtrain = xtrain';
xtest = xtest';

% process training set and validation set
xtrain_mean = sum(xtrain,1)/1024;
xtrain_var = var(xtrain,1,1);
xtrainbias = xtrain - xtrain_mean;
xtrainbias = xtrainbias./xtrain_var;
xtrainbias = xtrainbias';

xtest_mean = sum(xtest,1)/1024;
xtest_var = var(xtest,1,1);
xtestbias = xtest - xtest_mean;
xtestbias = xtestbias./xtest_var;
xtestbias = xtestbias';


epochs=1;
dim=size(xtrainbias,2);
num=size(xtrainbias,1);

% initial random
omegas=rand(1,dim);
lr=0.001;
sample=randperm(900);

acc_train(1)=1-sum(abs(((xtrainbias*omegas'>0)-ytrain)))/900;
acc_test(1)=1-sum(abs(((xtestbias*omegas'>0)-ytest)))/100;

while acc_train(epochs)~=1 && epochs<10000
    for j=sample
        y=sum(xtrainbias(j,:).*omegas);
        y=y>0;
        e=ytrain(j)-y;
        omegas=omegas+e*lr*xtrainbias(j,:);
    end
    epochs=epochs+1;
    acc_train(epochs)=1-sum(abs(((xtrainbias*omegas'>0)-ytrain)))/900;
    acc_test(epochs)=1-sum(abs(((xtestbias*omegas'>0)-ytest)))/100;
end
figure
plot(1:epochs,acc_train)
title('training accuracy(mean and variance processed)')
figure
plot(1:epochs,acc_test)
title('validation accuracy(mean and variance processed)')
