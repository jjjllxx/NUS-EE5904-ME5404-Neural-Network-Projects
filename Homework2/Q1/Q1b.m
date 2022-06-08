clc;clear;close all;
threshold=1e-6;
x(1)=rand*2-1; % initialize x between(-1,1)
y(1)=rand*2-1; % initialize y between(-1,1)
fxy(1)=(1-x(1))^2+100*(y(1)-x(1)^2)^2; % initial f(x,y)
iter=1;
learning_rate=0.001;
while fxy(iter)>threshold
    % update x
    x(iter+1)=x(iter)-learning_rate*(2*(x(iter)-1)+400*x(iter)*(x(iter)*x(iter)-y(iter)));
    % update y
    y(iter+1)=y(iter)-learning_rate*200*(y(iter)-x(iter)*x(iter));
    iter=iter+1;
    % update f(x,y)
    fxy(iter)=(1-x(iter))^2+100*(y(iter)-x(iter)^2)^2;
end
t=1:iter;

figure

subplot(121)
plot(t,x,'linewidth',2)
hold on
plot(t,y,'linewidth',2)
legend('x-value', 'y-value')
xlabel('iteration')
title(['x-value,y-value(learning rate=',num2str(learning_rate),')'])

subplot(122)
plot(t,fxy,'linewidth',2)
legend('f(x,y)-value')
xlabel('iteration')
title(['f(x,y)-value (learning rate=',num2str(learning_rate),')'])

sgtitle(['Gradient descent method(learning rate=',num2str(learning_rate),')'])
