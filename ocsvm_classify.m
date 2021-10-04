function [predictLabel,decValues]=ocsvm_classify(ocsvmModel,normParam,testData)

testData=bsxfun(@rdivide,...
    testData-repmat(normParam.min,size(testData,1),1),...
    normParam.max-normParam.min);
testLabel=ones(size(testData,1),1);

[predictLabel,~,decValues]=svmpredict(testLabel,testData,ocsvmModel);