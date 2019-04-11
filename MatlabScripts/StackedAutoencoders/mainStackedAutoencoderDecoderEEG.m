clear;
clc;
%% Loading data
featuresVectors= load ('C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\PhD\myPublications\EEGDecoding\DataSetMasterThesis\S1_data.mat');
featuresVectors=featuresVectors.Data;  
sizeFeatureVectors=size(featuresVectors);
labels=load('C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\PhD\myPublications\EEGDecoding\DataSetMasterThesis\S1_labels.mat');
labels=labels.Data;
% Divide the data into training and test sets
trainingDataDist=0.80;

[trainingData, testData] = DivideDataSetTrainTestStackedAutoencoderSTFT(featuresVectors,labels,trainingDataDist);
% Make dataCompatible for autoencoder

featuresVectorsTraining=trainingData.featureVectorsTraining;
labelsTraining=trainingData.labelsTraining;

[TrainingFeatVectCompStackedAutoencoder,TrainingLabelsCompStackedAutoencoder]=makeDataCompStackedAutoencoderSTFT(featuresVectorsTraining,labelsTraining);

hiddenSize1 = 900;

autoenc1 = trainAutoencoder(TrainingFeatVectCompStackedAutoencoder,hiddenSize1, ...
    'MaxEpochs',200, ...
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
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat3 = encode(autoenc3,feat2);


hiddenSize4 = 100;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat4 = encode(autoenc4,feat3);

% hiddenSize5=40;
% autoenc5 = trainAutoencoder(feat4,hiddenSize5, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% feat5=encode(autoenc5,feat4);

% hiddenSize6=15;
% autoenc6 = trainAutoencoder(feat5,hiddenSize6, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% feat6=encode(autoenc6,feat5);
% 
% hiddenSize7=5;
% autoenc7 = trainAutoencoder(feat6,hiddenSize7, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% feat7=encode(autoenc7,feat6);


softnet = trainSoftmaxLayer(feat4,TrainingLabelsCompStackedAutoencoder,'MaxEpochs',400);
stackednet = stack(autoenc1,autoenc2,autoenc3,autoenc4,softnet);
featVectTest=testData.featureVectorsTest;
labelsTest=testData.labelsTest;
[TestFeatVectCompStackedAutoencoder,TestLabelsCompStackedAutoencoder]=makeDataCompStackedAutoencoderSTFT(featVectTest,labelsTest);
[testFlattenFeatVectorsAutoenc] = flattenInputsStackedAutoencder(TestFeatVectCompStackedAutoencoder);
y = stackednet(testFlattenFeatVectorsAutoenc);
plotconfusion(TestLabelsCompStackedAutoencoder,y);

[trainFlattenFeatVectorsAutoenc] = flattenInputsStackedAutoencder(TrainingFeatVectCompStackedAutoencoder);

stackednetFinetuned = train(stackednet,trainFlattenFeatVectorsAutoenc,TrainingLabelsCompStackedAutoencoder);

y = stackednetFinetuned(testFlattenFeatVectorsAutoenc);
plotconfusion(TestLabelsCompStackedAutoencoder,y);





% Hyperaparamters for defined autoencoders and softmax layer
% autoenc1Params.hiddensize=600;
% autoenc1Params.MaxEpochs=50;
% autoenc1Params.L2WeightRegularization=0.004;
% autoenc1Params.SparsityRegularization=4;
% autoenc1Params.SparsityProportion=0.15;
% 
% autoenc2Params.hiddensize=500;
% autoenc2Params.MaxEpochs=100;
% autoenc2Params.L2WeightRegularization=0.002;
% autoenc2Params.SparsityRegularization=4;
% autoenc2Params.SparsityProportion=0.1;
% 
% 
% autoenc3Params.hiddensize=200;
% autoenc3Params.MaxEpochs=100;
% autoenc3Params.L2WeightRegularization=0.002;
% autoenc3Params.SparsityRegularization=4;
% autoenc3Params.SparsityProportion=0.1;
% 
% 
% autoenc4Params.hiddensize=100;
% autoenc4Params.MaxEpochs=100;
% autoenc4Params.L2WeightRegularization=0.002;
% autoenc4Params.SparsityRegularization=4;
% autoenc4Params.SparsityProportion=0.1;
% 
% autoenc5Params.hiddensize=40;
% autoenc5Params.MaxEpochs=100;
% autoenc5Params.L2WeightRegularization=0.002;
% autoenc5Params.SparsityRegularization=4;
% autoenc5Params.SparsityProportion=0.1;
% 
% 
% autoenc6Params.hiddensize=15;
% autoenc6Params.MaxEpochs=100;
% autoenc6Params.L2WeightRegularization=0.002;
% autoenc6Params.SparsityRegularization=4;
% autoenc6Params.SparsityProportion=0.1;
% 
% autoenc6Params.hiddensize=15;
% autoenc6Params.MaxEpochs=100;
% autoenc6Params.L2WeightRegularization=0.002;
% autoenc6Params.SparsityRegularization=4;
% autoenc6Params.SparsityProportion=0.1;
% 
% autoenc7Params.hiddensize=10;
% autoenc7Params.MaxEpochs=100;
% autoenc7Params.L2WeightRegularization=0.002;
% autoenc7Params.SparsityRegularization=4;
% autoenc7Params.SparsityProportion=0.1;
% 
% softmaxParams.MaxEpochs=100;
% %-- Train the defined architure stcked autoencoders
% trainedStackedNet=stackedAutoEncoderSTFTFeatures (xTrainImages,...
%                                                     tTrain,...
%                                                     autoenc1Params, autoenc2Params, autoenc3Params,...
%                                                     autoenc4Params, autoenc5Params,autoenc6Params,...
%                                                     autoenc7Params,softmaxParams);






