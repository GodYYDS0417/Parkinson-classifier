function [validationAccuracy,C,AUC,X1,Y1,X2,Y2,X3,Y3] = trainClassifier_KNN(trainingData)

response = trainingData(:,1);% label
trainingData(:,1) = [];
predictors = trainingData;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1; 2; 3]);


% Perform cross-validation
partitionedModel = crossval(classificationKNN, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% confusion matrix
C =  confusionmat(response,validationPredictions);

% ROC curve
[X1,Y1,T,AUC(1,:)] = perfcurve(response,validationScores(:,1),'1');
[X2,Y2,T,AUC(2,:)] = perfcurve(response,validationScores(:,2),'2');   
[X3,Y3,T,AUC(3,:)] = perfcurve(response,validationScores(:,3),'3');