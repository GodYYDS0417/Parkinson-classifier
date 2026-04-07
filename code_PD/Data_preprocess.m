clear;clc;

imagesize1 = [100 100 3];% 記得改size
imagesize2 = [100 100];% 記得改size ,還有customreader那邊
epoch = 50;

DatasetPath=fullfile('C:','Users/USER/Desktop/Meng-Ying/data_0803_clip');
trainDigitData=imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
trainDigitData.ReadFcn = @customreader;
%[trainDigitData,testData]=imds.splitEachLabel(0.8,0.2,'Randomize');

layers = [
    imageInputLayer(imagesize1,"Name","imageinput")
    convolution2dLayer(imagesize2,32,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    fullyConnectedLayer(3,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
opts = trainingOptions("sgdm",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",4,...
    "MiniBatchSize",16,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",50);

alltraindata = trainDigitData.readall;
[rnumtrain , cnumtrain] = size(trainDigitData.Files);
for i=1:rnumtrain
    store = cell2mat(alltraindata(i,1));
    [a ,b ,c ] = size(store);
    data(i,:) = double(reshape(store,1,a*b*c));          
end
traindata_label = [double(trainDigitData.Labels) ,data];


%%
for i=1:epoch
    i
    % Classification Learner
    %-------  SVM -------%
    [Accuracy_SVM(i), Confusionmat_SVM, AUC_SVM(i,:), X0_SVM, Y0_SVM,X1_SVM,Y1_SVM,X2_SVM,Y2_SVM] = trainClassifier_SVM(traindata_label);
    
    %-------  KNN -------%
    [Accuracy_KNN(i), Confusionmat_KNN, AUC_KNN(i,:), X0_KNN, Y0_KNN,X1_KNN,Y1_KNN,X2_KNN,Y2_KNN] = trainClassifier_KNN(traindata_label);
    
    %------- Linear Discriminant -------%
    [Accuracy_LD(i),  Confusionmat_LD,  AUC_LD(i,:),  X0_LD , Y0_LD, X1_LD, Y1_LD, X2_LD, Y2_LD ] = trainClassifier_LD (traindata_label);
    
    %------- Tree -------%
    [Accuracy_Tree(i),Confusionmat_Tree,AUC_Tree(i,:),X0_Tree,Y0_Tree,X1_Tree,Y1_Tree,X2_Tree,Y2_Tree] = trainClassifier_Tree(traindata_label);
    
    %------- Gaussian Naive Bayes -------%
    [Accuracy_NB(i), Confusionmat_NB, AUC_NB(i,:), X0_NB, Y0_NB, X1_NB, Y1_NB, X2_NB, Y2_NB] = trainClassifier_NB(traindata_label);
    
    % Neural Network
    [net ,info] = trainNetwork(trainDigitData,layers,opts);
    [YPred,YScore] = classify(net,trainDigitData);
    Accuracy_NN(i) = mean(YPred == trainDigitData.Labels);
    Confusionmat_NN = confusionmat(trainDigitData.Labels,YPred);
    
    [X0_NN,Y0_NN,T,AUC_NN(i,1)] = perfcurve(trainDigitData.Labels,YScore(:,1),'0');
    [X1_NN,Y1_NN,T,AUC_NN(i,2)] = perfcurve(trainDigitData.Labels,YScore(:,2),'1');   
    [X2_NN,Y2_NN,T,AUC_NN(i,3)] = perfcurve(trainDigitData.Labels,YScore(:,3),'2');
    
    
    % F1-score
    f1_score_KNN(i,:)  = f1_score(Confusionmat_KNN);
    f1_score_SVM(i,:)  = f1_score(Confusionmat_SVM);
    f1_score_LD(i,:)   = f1_score(Confusionmat_LD);
    f1_score_Tree(i,:) = f1_score(Confusionmat_Tree);
    f1_score_NB(i,:)   = f1_score(Confusionmat_NB);
    f1_score_NN(i,:)   = f1_score(Confusionmat_NN);
    %F1_score = [f1_score_SVM;f1_score_LD;f1_score_KNN;f1_score_NN];

end

%% 
Accuracy = [Accuracy_Tree;Accuracy_SVM;Accuracy_KNN;Accuracy_LD;Accuracy_NB;Accuracy_NN];
F1_score_0 = [f1_score_Tree(:,1),f1_score_SVM(:,1),f1_score_KNN(:,1),f1_score_LD(:,1),f1_score_NB(:,1),f1_score_NN(:,1)];
F1_score_1 = [f1_score_Tree(:,2),f1_score_SVM(:,2),f1_score_KNN(:,2),f1_score_LD(:,2),f1_score_NB(:,2),f1_score_NN(:,2)];
F1_score_2 = [f1_score_Tree(:,3),f1_score_SVM(:,3),f1_score_KNN(:,3),f1_score_LD(:,3),f1_score_NB(:,3),f1_score_NN(:,3)];

AUC_0 = [AUC_Tree(:,1), AUC_SVM(:,1), AUC_KNN(:,1), AUC_LD(:,1), AUC_NB(:,1), AUC_NN(:,1)];
AUC_1 = [AUC_Tree(:,2), AUC_SVM(:,2), AUC_KNN(:,2), AUC_LD(:,2), AUC_NB(:,2), AUC_NN(:,2)];
AUC_2 = [AUC_Tree(:,3), AUC_SVM(:,3), AUC_KNN(:,3), AUC_LD(:,3), AUC_NB(:,3), AUC_NN(:,3)];

writedata();
plot_all_figure();%  accuracy*1, f1-score*3, AUC*3 and ROC curve*3.

%% Neural Network plot
Accuracy_NN_1 = info.TrainingAccuracy;
Loss_NN = info.TrainingLoss;
writematrix(Accuracy_NN_1,'Accuracy_Datapreprocess.xlsx','Range','A1');
writematrix(Loss_NN,'Loss_Datapreprocess.xlsx','Range','A1');
