clc;clear;close all;
xtrain=-1.6:0.05:1.6;
xtest=-3:0.01:3;
ytrain=sin(1.2*pi*xtrain)-cos(2.4*pi*xtrain);
ytest=sin(1.2*pi*xtest)-cos(2.4*pi*xtest);
n=[1:10,20,50,100]; % different structure of layers
trainmode='trainbr'; % set train mode here
for i=n
    
    net = feedforwardnet(i,trainmode);
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net = configure(net,xtrain,ytrain);
    net.trainparam.lr=0.01;
    net.trainparam.epochs=10000;
    net.trainparam.goal=1e-8;
    net.divideParam.trainRatio=1.0;
    net.divideParam.valRatio=0.0; 
    net.divideParam.testRatio=0.0; 
    
    [net,tr]=train(net,xtrain,ytrain);
    ypred=sim(net,xtest);
    
    figure
    plot(xtest,ytest,'LineWidth',2)
    hold on
    plot(xtrain,ytrain,'o')
    hold on
    plot(xtest,ypred,'LineWidth',2)
    ylim([-2 2.5])
    legend('Original function','Train Points','MLP result')
    title([trainmode,' mode(Hidden neurons: ',num2str(i),')'])
    ylabel('y-value')
    xlabel('x-value')
end