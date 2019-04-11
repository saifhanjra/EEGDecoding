function trainedStackedNet=stackedAutoEncoderSTFTFeatures (TrainingFeatVectCompStackedAutoencoder,TrainingLabelsCompStackedAutoencoder,autoenc1Params, autoenc2Params, autoenc3Params, autoenc4Params, autoenc5Params,autoenc6Params, autoenc7Params,softmaxParams)


autoenc1 = trainAutoencoder(TrainingFeatVectCompStackedAutoencoder,autoenc1Params.hiddensize, ...
    'MaxEpochs',autoenc1Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc1Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc1Params.SparsityRegularization, ...
    'SparsityProportion',autoenc1Params.SparsityProportion, ...
    'ScaleData', false);

feat1 = encode(autoenc1,TrainingFeatVectCompStackedAutoencoder);


autoenc2 = trainAutoencoder(feat1,autoenc2Params.hiddensize, ...
    'MaxEpochs',autoenc2Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc2Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc2Params.SparsityRegularization, ...
    'SparsityProportion',autoenc2Params.SparsityProportion, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

autoenc3 = trainAutoencoder(feat2,autoenc3Params.hiddensize, ...
    'MaxEpochs',autoenc3Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc3Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc3Params.SparsityRegularization, ...
    'SparsityProportion',autoenc3Params.SparsityProportion, ...
    'ScaleData', false);

feat3=encode(autoenc3,feat2);

autoenc4 = trainAutoencoder(feat3,autoenc4Params.hiddensize, ...
    'MaxEpochs',autoenc4Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc4Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc4Params.SparsityRegularization, ...
    'SparsityProportion',autoenc4Params.SparsityProportion, ...
    'ScaleData', false);


feat4=encode(autoenc4, feat3);


autoenc5 = trainAutoencoder(feat4,autoenc5Params.hiddensize, ...
    'MaxEpochs',autoenc5Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc5Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc5Params.SparsityRegularization, ...
    'SparsityProportion',autoenc5Params.SparsityProportion, ...
    'ScaleData', false);

feat5=encode(autoenc5,feat4);

autoenc6 = trainAutoencoder(feat5,autoenc6Params.hiddensize, ...
    'MaxEpochs',autoenc6Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc6Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc6Params.SparsityRegularization, ...
    'SparsityProportion',autoenc6Params.SparsityProportion, ...
    'ScaleData', false);


feat6=encode(autoenc6,feat5);

autoenc7 = trainAutoencoder(feat6,autoenc7Params.hiddensize, ...
    'MaxEpochs',autoenc7Params.MaxEpochs, ...
    'L2WeightRegularization',autoenc7Params.L2WeightRegularization, ...
    'SparsityRegularization',autoenc7Params.SparsityRegularization, ...
    'SparsityProportion',autoenc7Params.SparsityProportion, ...
    'ScaleData', false);

feat7=encode(autoenc7,feat6);

%---------------------Softmax Layer----------------------------------------

softnet = trainSoftmaxLayer(feat7,TrainingLabelsCompStackedAutoencoder,'MaxEpochs',400);

trainedStackedNet=stack(autoenc1,autoenc2,autoenc3,autoenc4,autoenc5,autoenc6,autoenc7,softnet);




end