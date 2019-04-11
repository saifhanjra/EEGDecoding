clear;
clc;
%--------------------------------------


%% Loading data
featuresVectors= load ('C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\PhD\myPublications\EEGDecoding\DataSetMasterThesis\S4_data.mat');
featuresVectors=featuresVectors.Data;  
sizeFeatureVectors=size(featuresVectors);
labels=load('C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\PhD\myPublications\EEGDecoding\DataSetMasterThesis\S4_labels.mat');
labels=labels.Data;
% Divide the data into training and test sets
trainingDataDist=0.80;

[trainingData, testData] = DivideDataSetTrainTestStackedAutoencoderSTFT(featuresVectors,labels,trainingDataDist);
% Make dataCompatible for autoencoder

featuresVectorsTraining=trainingData.featureVectorsTraining;
labelsTraining=trainingData.labelsTraining;
load 'trainedNet_CNN_STFT_S4.mat'
%------------------------------------------------------------------------------------------------
[TrainingFeatVectCompStackedAutoencoder,TrainingLabelsCompStackedAutoencoder]= extractConvFeatureMaps(trainedNet,featuresVectorsTraining,labelsTraining);



hiddenSize1 = 900;

autoenc1 = trainAutoencoder(TrainingFeatVectCompStackedAutoencoder,hiddenSize1, ...
    'MaxEpochs',300, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat1 = encode(autoenc1,TrainingFeatVectCompStackedAutoencoder);

hiddenSize2 = 500;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);


hiddenSize3 = 200;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat3 = encode(autoenc3,feat2);


hiddenSize4 = 100;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat4 = encode(autoenc4,feat3);


softnet = trainSoftmaxLayer(feat4,TrainingLabelsCompStackedAutoencoder,'MaxEpochs',400);
stackednet = stack(autoenc1,autoenc2,autoenc3,autoenc4,softnet);

featVectTest=testData.featureVectorsTest;
labelsTest=testData.labelsTest;
[TestFeatVectCompStackedAutoencoder,TestLabelsCompStackedAutoencoder]= extractConvFeatureMaps(trainedNet,featVectTest,labelsTest);
[testFlattenFeatVectorsAutoenc] = flattenInputsStackedAutoencder(TestFeatVectCompStackedAutoencoder);
y = stackednet(testFlattenFeatVectorsAutoenc);
plotconfusion(TestLabelsCompStackedAutoencoder,y);

[trainFlattenFeatVectorsAutoenc] = flattenInputsStackedAutoencder(TrainingFeatVectCompStackedAutoencoder);

stackednetFinetuned = train(stackednet,trainFlattenFeatVectorsAutoenc,TrainingLabelsCompStackedAutoencoder);
y = stackednetFinetuned(testFlattenFeatVectorsAutoenc);
plotconfusion(TestLabelsCompStackedAutoencoder,y);




%---------------------------------------------------------------------------
% layersCNN=trainedNet.Layers;
% layersCNNTransfered=trainedNet.Layers(1:2);
% 
% newLayers = [ ...
%     layersCNNTransfered
%     fullyConnectedLayer(900);
%     softmaxLayer
%     
%     fullyConnectedLayer(2);
%     softmaxLayer
%     classificationLayer
%     ];
