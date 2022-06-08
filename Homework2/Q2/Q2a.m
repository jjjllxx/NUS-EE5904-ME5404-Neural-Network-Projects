clc;clear;close all;
xtrain=-1.6:0.05:1.6;
xtest=-3:0.01:3;
ytrain=sin(1.2*pi*xtrain)-cos(2.4*pi*xtrain);
ytest=sin(1.2*pi*xtest)-cos(2.4*pi*xtest);
n=[1:10,20,50,100]; % different structure of layers
epoches=300;

for i=n
    [mynet]=train_seq(i,xtrain,ytrain,epoches);
    ypred=mynet(xtest);
    figure
    plot(xtest,ytest,'LineWidth',2)
    hold on
    plot(xtrain,ytrain,'o')
    hold on
    plot(xtest,ypred,'LineWidth',2)
    ylim([-2 2.5])
    legend('Original function','Train Points','MLP result')
    title(['Sequentional Mode(Epochs: ',num2str(epoches),' Hidden neurons: ',num2str(i),')'])
    ylabel('y-value')
    xlabel('x-value')
end

% according to the provided code
function [net] = train_seq(n,x,y,epochs)

% Construct a 1-n-1 MLP and conduct sequential training.
train_num=size(x,2);

% 1. Change the input to cell array form for sequential training
x_c = num2cell(x, 1);
y_c = num2cell(y, 1);

% 2. Construct and configure the MLP
net = fitnet(n);
net.divideFcn = 'dividetrain'; % input for training only
net.divideParam.trainRatio=1.0;
net.divideParam.valRatio=0.0; 
net.divideParam.testRatio=0.0; 
net.trainParam.epochs = epochs;

% 3. Train the network in sequential mode
for i = 1 : epochs
    display(['Neurons: ',num2str(n),' Epoch: ', num2str(i)])
    idx = randperm(train_num); % shuffle the input
    net = adapt(net, x_c(:,idx), y_c(:,idx));
end

end