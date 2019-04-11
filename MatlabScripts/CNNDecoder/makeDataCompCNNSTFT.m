function [featVectCompCNN,labelsCompCNN]=makeDataCompCNNSTFT(featuresVectors,labels)
totalFeatureVectors=numel(labels);
% featVectCompCNN=cell(1,totalFeatureVectors);
for i=1:totalFeatureVectors
    featVectCompCNN_i=featuresVectors(i,:,:);
    featVectCompCNN_i=reshape(featVectCompCNN_i, [93,32,1]);
    featVectCompCNN(:,:,1,i)=featVectCompCNN_i;
%     featVectCompCNN{1,i}=featVectCompCNN_i;
    
end
labelsCompCNN=categorical(labels);

end

