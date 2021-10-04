clear all;close all;clc;
addpath(genpath(pwd));

% Data
load train_data;
trainData=trainData(:,2:3);
normParam.min=min(trainData);
normParam.max=max(trainData);

trainData=bsxfun(@rdivide,...
    trainData-repmat(min(trainData),size(trainData,1),1),...
    max(trainData)-min(trainData));

trainData=consolidator(trainData,[],@mean,1e-2);
trainLabel=ones(size(trainData,1),1);

% DFN
s=ocsvm_dfn(trainData);

% OCSVM
mat2svm([trainLabel trainData]);
[trainLabel,trainData]=libsvmread('mySVMdata.txt');
options=sprintf('-s 2 -n 0.0001 -g %f',1/2/s^2);
ocsvmModel=svmtrain(trainLabel,trainData,options);

save ocsvm_model ocsvmModel normParam;

%% Threshold Adjustment
clear all;close all;clc;
load ocsvm_model;
testData=repmat(normParam.min-10,1e6,1)+...
    bsxfun(@times,rand(1e6,2),(normParam.max-normParam.min+20));
[predictLabel,decValues]=ocsvm_classify(ocsvmModel,normParam,testData);

figure;clf;
plot(ocsvmModel.SVs(:,1)*(normParam.max(1)-normParam.min(1))+normParam.min(1),...
     ocsvmModel.SVs(:,2)*(normParam.max(2)-normParam.min(2))+normParam.min(2),'ro','MarkerSize',5);
hold on;
load train_data;
trainData=trainData(:,2:3);
plot(trainData(:,1),trainData(:,2),'g.','MarkerSize',3);
hold on;
plot(testData(abs(decValues)<=1e-4,1),testData(abs(decValues)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+1e-2)<=1e-4,1),testData(abs(decValues+1e-2)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+2e-2)<=1e-4,1),testData(abs(decValues+2e-2)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+3e-2)<=1e-4,1),testData(abs(decValues+3e-2)<=1e-4,2),'k.','MarkerSize',3);

supportVector=ocsvmModel.SVs.*(normParam.max-normParam.min)+normParam.min;
boundaryData0=testData(abs(decValues)<=1e-4,:);
boundaryData1=testData(abs(decValues+1e-2)<=1e-4,:);
boundaryData2=testData(abs(decValues+2e-2)<=1e-4,:);
boundaryData3=testData(abs(decValues+3e-2)<=1e-4,:);
save data_2d supportVector trainData boundaryData0 boundaryData1 boundaryData2 boundaryData3;

%% Time Validation
clear all;close all;clc;
load ocsvm_model;
% ocsvmModel.rho=ocsvmModel.rho-2e-2;

clear ibrlData;
load ibrl_data;
index=date*24*60*60+time;
ibrlData=[month index moteid temperature humidity];
ibrlData(ibrlData(:,1)~=3,:)=[];  

% 
clear predictLabel moteData;
for i=37
    moteData{i}=ibrlData(ibrlData(:,3)==i,[2 4 5]);
    predictLabel{i}=ocsvm_classify(ocsvmModel,normParam,moteData{i}(:,[2 3]));
    figure(i);clf;
    plot(moteData{i}(:,[2 3]),'b-');
    hold on;
    plot(find(predictLabel{i}==-1),moteData{i}(predictLabel{i}==-1,[2 3]),...
        'ro','linewidth',2);
end

moteData=moteData{i}(:,[2 3]);
anomalyIndex=find(predictLabel{i}==-1);
anomalyData=moteData(anomalyIndex,:);
save timeData37 moteData anomalyIndex anomalyData;

%% Data validation
% Labelling
clear all;close all;clc;
load ibrl_data;

ibrlData=[month date time moteid temperature humidity];
ibrlData(:,[1 2 3])=[]; 

for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
end

I{1}=[1 2411 2417 2501 2504 2509 2520 2521 2534 2.26e4:2.38e4];
J{1}=[4e4:4.3e4];

I{2}=[1 276 4602 4611 4613 4628 4630 1.88e4:1.98e4];
J{2}=[4.25e4:4.65e4];

I{33}=[1.42e4:1.48e4];
J{33}=[3.3e4:3.48e4];

I{35}=[4.5e4:4.55e4];
J{35}=[3.5e4:3.6e4];

I{37}=unique(max(2.046e4,min(2.36e4,find(moteData{37}(:,1)>37))));
I{37}(1)=[];I{37}(end)=[];
J{37}=[4.7e4:4.74e4];

negativeData=[];
positiveData=[];
for i=[1 2 33 35 37]
    negativeData=[negativeData;moteData{i}(I{i},:)];
    positiveData=[positiveData;moteData{i}(J{i},:)];
    figure(i)
    plot(moteData{i},'b-');hold on;
    plot(I{i},moteData{i}(I{i},:),'ro');hold on;
    plot(J{i},moteData{i}(J{i},:),'go');
end

save test_data positiveData negativeData;

% Validation
load ocsvm_model;
ocsvmModel.rho=ocsvmModel.rho-3e-2;

testData=[positiveData;negativeData];
trueLabel=[ones(size(positiveData,1),1);-1*ones(size(negativeData,1),1)];
[predictLabel,boundaryLabel]=ocsvm_classify(ocsvmModel,normParam,testData);

figure;clf;
plot(positiveData(:,1),positiveData(:,2),'b*');hold on;
plot(negativeData(:,1),negativeData(:,2),'r*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);

DR=length(find(predictLabel==-1 & trueLabel==-1))/size(negativeData,1);
FNR=1-DR;
FPR=length(find(predictLabel==-1 & trueLabel==1))/size(positiveData,1);

%%
n=size(trainData,1);
for i=1:n
    X1=trainData;
    X1(i,:)=[];
    [~,D] = knnsearch(X1,trainData(i,:));
    dmin(i)=D;
    dmaxi=0;
    for j=1:n
        dmaxij=norm(trainData(i,:)-trainData(j,:));
        if dmaxij>dmaxi
           dmaxi=dmaxij;
        end
    end
    dmax(i)=dmaxi;
end

s=(1e-5:1e-2:1)';
f=zeros(length(s),1);
for i=1:length(s)
    [f(i),~]=obj_fcn(dmin,dmax,s(i));
end

save dfn_data Js s f;

