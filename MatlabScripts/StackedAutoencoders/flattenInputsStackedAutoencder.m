function [testFlattenFeatVectorsAutoenc] = flattenInputsStackedAutoencder(TestFeatVectCompStackedAutoencoder)

[featVectHeight, featVectwidth]=size(TestFeatVectCompStackedAutoencoder{1, 1});

inputSize=featVectwidth*featVectHeight;

testFlattenFeatVectorsAutoenc=zeros(inputSize,numel(TestFeatVectCompStackedAutoencoder));
for i=1:numel(TestFeatVectCompStackedAutoencoder)
     testFlattenFeatVectorsAutoenc(:,i)=TestFeatVectCompStackedAutoencoder{i}(:);
end

end