clear;clc;
train_num=randperm(900);
xtrain=zeros(900,1024);xtest=zeros(100,1024);

for i=1:450
    now=num2str(i-1);
    xtrain(i,:)=reshape(rgb2gray(imread(['group_0/cat/',repmat('0',1,3-length(now)),now,'.jpg'])),1,1024);
    xtrain(i+450,:)=reshape(rgb2gray(imread(['group_0/airplane/',repmat('0',1,3-length(now)),now,'.jpg'])),1,1024);
end
for i=451:500
    now=num2str(i-1);
    xtest(i-450,:)=reshape(rgb2gray(imread(['group_0/cat/',repmat('0',1,3-length(now)),now,'.jpg'])),1,1024);
    xtest(i-400,:)=reshape(rgb2gray(imread(['group_0/airplane/',repmat('0',1,3-length(now)),now,'.jpg'])),1,1024);
end