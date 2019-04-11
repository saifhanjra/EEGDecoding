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

[featVectCompCNN,labelsCompCNN]=makeDataCompCNNSTFT(featuresVectorsTraining,labelsTraining);

featuresVectorsValidation=testData.featureVectorsTest;
labelsValidation=testData.labelsTest;

[featVectValidationCompCNN,labelsValidationCompCNN]=makeDataCompCNNSTFT(featuresVectorsValidation,labelsValidation);

layers = [ ...
    imageInputLayer([93 32 1])
    
    convolution2dLayer([93,1],30,'Stride',[1,1])
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer
    maxPooling2dLayer([1,3],'Stride',1, 'Padding', [0,0])
    
%     fullyConnectedLayer(900);
%     softmaxLayer
    
    fullyConnectedLayer(2);
    softmaxLayer
    classificationLayer
    ];
    
options = trainingOptions('sgdm',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',5,...
    'L2Regularization',0.001,...
    'MaxEpochs',400,...
    'MiniBatchSize',100,...
    'ValidationData',{featVectValidationCompCNN,labelsValidationCompCNN},...
    'ValidationFrequency',50,...
    'Verbose',true,...
    'Plots','training-progress');

[trainedNet, trainInfo] = trainNetwork(featVectCompCNN,labelsCompCNN,layers,options);
