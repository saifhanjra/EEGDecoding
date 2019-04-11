function [featVectCompStackedAutoencoder,labelsCompStackedAutoencoder]= extractConvFeatureMaps(trainedNet,featuresVectors,labels) 
[featVectCompCNN,~]=makeDataCompCNNSTFT(featuresVectors,labels);
%------------------------------------------------------------------------------------------------

totalTrainingExamples=numel(labels);
featVectCompStackedAutoencoder=cell(1,totalTrainingExamples);

[featMapConvHeight,featMapConvWidth, featMapConvDepth]=size(activations(trainedNet,featVectCompCNN(:,:,1,1),2));
featMapConvSize=featMapConvHeight*featMapConvWidth*featMapConvDepth;

for i=1:totalTrainingExamples
    featMapCov_i=activations(trainedNet,featVectCompCNN(:,:,1,i),2);
    featMapCov_i=reshape(featMapCov_i,[featMapConvSize,1]);
    featVectCompStackedAutoencoder{1,i}=featMapCov_i;
end
labelsCompStackedAutoencoder=zeros(2,totalTrainingExamples);

labelsClass_1=find(labels==1);
labelsCompStackedAutoencoder(1,labelsClass_1)=1;

labelsClass_0=find(labels==0);
labelsCompStackedAutoencoder(2,labelsClass_0)=1;

end
