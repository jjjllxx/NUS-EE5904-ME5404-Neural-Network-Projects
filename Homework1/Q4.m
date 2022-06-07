clear;clc;close all;
x=[0, 0.8, 1.6, 3, 4, 5;1, 1, 1, 1, 1, 1];
y=[0.5, 1, 4, 5, 6, 8];

% LLS
omega1=(x*x')^(-1)*x*y';
t=-1:0.01:7;
ft1=omega1(1)*t+omega1(2);
figure
plot(t,ft1,'LineWidth',2)
hold on
plot(x(1,:),y,'o','MarkerSize',10)
legend('LLS results','pairs')
title('LLS method')


% % LMS
rate=0.01;epochs=100;
omega2=zeros(epochs+1,2);
omega2(1,:)=rand(1,2);

for i=1:epochs
    e=y-omega2(i,:)*x;
    omega2(i+1,:)=omega2(i,:)+rate*e*x';
end
ft2=omega2(epochs+1,1)*t+omega2(epochs+1,2);

figure
plot(t,ft2,'LineWidth',2)
hold on
plot(x(1,:),y,'o','MarkerSize',10)
legend('LMS results','pairs')
title('LMS method')

figure
t=1:101;
plot(t,omega2(:,1),'LineWidth',2)
hold on
plot(t,omega2(:,2),'LineWidth',2)
title('LMS weight trajectory(learning rate=0.01)')
legend('w','b')

% figure
% rates=[0.005,0.01,0.05,0.1];
% pos=1;
% for rate=rates
%     for i=1:epochs
%         e=y-omega2(i,:)*x;
%         omega2(i+1,:)=omega2(i,:)+rate*e*x';
%     end
%     ft2=omega2(epochs+1,1)*t+omega2(epochs+1,2);
%     
%     subplot(size(rates,2),1,pos)
%     t=1:101;
%     plot(t,omega2(:,1),'LineWidth',2)
%     hold on
%     plot(t,omega2(:,2),'LineWidth',2)
%     title(['LMS weight trajectory(learning rate=',num2str(rate),')'])
%     legend('w','b')
%     pos=pos+1;
% end
