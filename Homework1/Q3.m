clear;close all;clc;
rate=[0.01,0.1,1,5];
epoches=100;
x1=[0,0,1,1;0,1,0,1];

%AND
y1=[0,0,0,1];
offline1=[-1.5,1,1];
omega1=training(x1,y1,rate,epoches,offline1,'AND');

%OR
y2=[0,1,1,1];
offline2=[-0.5,1,1];
omega2=training(x1,y2,rate,epoches,offline2,'OR');

%COMPLEMENT
x2=[0,1];
y3=[1,0];
offline3=[0.5,-1];
omega3=training(x2,y3,rate,epoches,offline3,'COMPLEMENT');

%NAND
y4=[1,1,1,0];
offline4=[1.5,-1,-1];
omega4=training(x1,y4,rate,epoches,offline4,'NAND');

%EXCLUSIVE OR
y5=[0,1,1,0];
offline5=[];
omega5=training(x1,y5,rate,epoches,offline5,'XOR');


function [omega]=training(x,y,rate,epoches,offline,name)
dim=size(x,1);
num=size(x,2);
omegas=zeros(epoches+1,dim+1);
% initial random
omegas(1,:)=rand(1,dim+1);

figure
% learning procedure
for k=1:size(rate,2)
    r=rate(k);
    for t=1:epoches
        pos=mod(t-1,num)+1;
        now=[1,x(:,pos)'];
        d=sum(now.*omegas(t,:));
        d=d>0;
        e=d-y(pos);
        omegas(t+1,:)=omegas(t,:)-e*r*now;
    end
    t=0:epoches;
    subplot(size(rate,2),1,k)
    if dim==1
        plot(t,omegas(:,1)')
        hold on
        plot(t,omegas(:,2)')
        legend('b','w1')
    elseif dim==2
        plot(t,omegas(:,1)')
        hold on
        plot(t,omegas(:,2)')
        hold on
        plot(t,omegas(:,3)')
        legend('b','w1','w2')
    end
    title([name,'(learning rate=',num2str(r),')'])
end
omega=omegas(epoches+1,:);

% draw point and decision boundary
figure
if dim==2
    if ~isempty(offline)
        subplot(121)
        drawDecBou(x,y,offline)
        title([name,' decision boundary(offline)'])
    end
    if ~isempty(offline)
        subplot(122)
    end
    drawDecBou(x,y,omega)
    title([name,' decision boundary(learning)'])
    
elseif dim==1
    subplot(121)
    drawDecBou(x,y,offline)
    title([name,' decision boundary(offline)'])
   
    subplot(122)
    drawDecBou(x,y,omega)
    title([name,' decision boundary(learning)'])
end

end

% draw decision boundary
function []=drawDecBou(x,y,omega)
dim=size(x,1);
num=size(x,2);
t=-1:0.01:2;

if dim==2
    for i=1:num
        if (y(i))
            plot(x(1,i),x(2,i),'r*','MarkerSize',10);
        else
            plot(x(1,i),x(2,i),'go','MarkerSize',10);
        end
        hold on
    end
    o1=omega(1);o2=omega(2);o3=omega(3);
    ft=-(t*o2+o1)/o3;
    plot(t,ft,'LineWidth',2);
        
elseif dim==1
    for i=1:num
        if (y(i))
            plot(x(i),0,'r*','MarkerSize',10);
        else
            plot(x(i),0,'go','MarkerSize',10);
        end
        hold on
    end
    ft=ones(1,size(t,2))*omega(1);
    plot(ft,t,'LineWidth',2);
end

    xlim([-0.5,1.5]);
    ylim([-0.5,1.5]);
    axis square
    
end
