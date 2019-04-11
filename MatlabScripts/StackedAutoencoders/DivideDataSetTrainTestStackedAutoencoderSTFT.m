function[trainingData, testData] = DivideDataSetTrainTestStackedAutoencoderSTFT(featuresVectors,labels,trainingDataDist)

totalFeatureVectors=numel(labels);

indClass_1= find(labels==1);
indClass_0=find(labels==0);


featuresVectorsClass_1=featuresVectors(indClass_1,:,:);
featuresVectorsClass_0=featuresVectors(indClass_0,:,:);

labelsClass_1=ones(1,numel(indClass_1));
labelsClass_0=zeros(1,numel(indClass_0));

 

totalTrainingExamples=floor(trainingDataDist*totalFeatureVectors);
totalTrainingExamplesClass_1=floor(totalTrainingExamples/2);
totalTrainingExamplesClass_0=totalTrainingExamplesClass_1;

featureVectorsTrainingClass_1=featuresVectorsClass_1(1:totalTrainingExamplesClass_1,:,:);%for training examples of class 1
featureVectorsTestClass_1=featuresVectorsClass_1(totalTrainingExamplesClass_1+1:end,:,:);%for test examples of class 0

labelsTrainingClass_1=ones(1,totalTrainingExamplesClass_1);% labeling training example of class 1
labelsTestClass_1=ones(1,numel(labelsClass_1)-totalTrainingExamplesClass_1);%labeling test examples of class 0

featureVectorsTrainingClass_0=featuresVectorsClass_0(1:totalTrainingExamplesClass_0,:,:);%for training exaples of class 0
featureVectorsTestClass_0=featuresVectorsClass_0(totalTrainingExamplesClass_0+1:end,:,:);% for test exmaples of class 0

labelsTrainingClass_0=zeros(1,totalTrainingExamplesClass_0);% labeling training examples of class 0
labelsTestClass_0=zeros(1,numel(labelsClass_0)-totalTrainingExamplesClass_0);% labeling test examples of class 0


 featureVectorsTraining=[featureVectorsTrainingClass_1; featureVectorsTrainingClass_0];% training dataset examples
 labelsTraining=[labelsTrainingClass_1,labelsTrainingClass_0];%training dataset labels
 
 shuffleIndicesTraining=randperm(numel(labelsTraining));% genarate random numbers to shuffle the data1 
 
featureVectorsTraining=featureVectorsTraining(shuffleIndicesTraining,:,:);
labelsTraining=labelsTraining(1,shuffleIndicesTraining);

trainingData.featureVectorsTraining=featureVectorsTraining;
trainingData.labelsTraining=labelsTraining;


featureVectorsTest=[featureVectorsTestClass_1;featureVectorsTestClass_0];
labelsTest=[labelsTestClass_1, labelsTestClass_0];
shuffleIndicesTest=randperm(numel(labelsTest));
featureVectorsTest=featureVectorsTest(shuffleIndicesTest,:,:);
labelsTest=labelsTest(1,shuffleIndicesTest);

testData.featureVectorsTest=featureVectorsTest;
testData.labelsTest=labelsTest;
end