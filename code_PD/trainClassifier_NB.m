function [validationAccuracy ,C,AUC,X1,Y1,X2,Y2,X3,Y3] = trainClassifier_NB(trainingData)

response = trainingData(:,1);% label
trainingData(:,1) = [];
predictors = trainingData;

distributionNames =  repmat({'Normal'}, 1, length(predictors));

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [1; 2; 3]);
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [1; 2; 3]);
end

% Perform cross-validation
partitionedModel = crossval(classificationNaiveBayes, 'KFold', 5);

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