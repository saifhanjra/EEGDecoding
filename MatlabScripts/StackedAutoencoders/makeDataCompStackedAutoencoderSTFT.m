function [featVectCompStackedAutoencoder,labelsCompStackedAutoencoder]=makeDataCompStackedAutoencoderSTFT(featuresVectors,labels)
totalFeatureVectors=numel(labels);
featVectCompStackedAutoencoder=cell(1,totalFeatureVectors);
for i=1:totalFeatureVectors
    featVectCompStackedAutoencoder_i=featuresVectors(i,:,:);
    featVectCompStackedAutoencoder{1,i}=reshape(featVectCompStackedAutoencoder_i, [93,32]);
end

labelsCompStackedAutoencoder=zeros(2,totalFeatureVectors);

labelsClass_1=find(labels==1);
labelsCompStackedAutoencoder(1,labelsClass_1)=1;

labelsClass_0=find(labels==0);
labelsCompStackedAutoencoder(2,labelsClass_0)=1;

end