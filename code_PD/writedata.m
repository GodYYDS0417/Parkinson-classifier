%-------- output to excel -------%
%% Accuracy
writematrix(Accuracy' ,'C:\Users\USER\Desktop\Meng-Ying\Output\Accuracy.xlsx');

%% F1-score
writematrix(F1_score_0,'C:\Users\USER\Desktop\Meng-Ying\Output\F1-score_0.xlsx');
writematrix(F1_score_1,'C:\Users\USER\Desktop\Meng-Ying\Output\F1-score_1.xlsx');
writematrix(F1_score_2,'C:\Users\USER\Desktop\Meng-Ying\Output\F1-score_2.xlsx');

%% ROC
% KNN
ROC_X_KNN = [X0_KNN,X1_KNN,X2_KNN];
ROC_Y_KNN = [Y0_KNN,Y1_KNN,Y2_KNN];
writematrix(ROC_X_KNN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_KNN_X.xlsx','Sheet',1);
writematrix(ROC_Y_KNN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_KNN_Y.xlsx','Sheet',1);

% LD
ROC_X_LD = [X0_LD,X1_LD,X2_LD];
ROC_Y_LD = [Y0_LD,Y1_LD,Y2_LD];
writematrix(ROC_X_LD,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_LD_X.xlsx','Sheet',1);
writematrix(ROC_Y_LD,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_LD_Y.xlsx','Sheet',1);

% SVM
ROC_X_SVM = [X0_SVM,X1_SVM,X2_SVM];
ROC_Y_SVM = [Y0_SVM,Y1_SVM,Y2_SVM];
writematrix(ROC_X_SVM,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_SVM_X.xlsx','Sheet',1);
writematrix(ROC_Y_SVM,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_SVM_Y.xlsx','Sheet',1);

% NB
writematrix(X0_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_X.xlsx','Sheet',1,'Range','A1');
writematrix(X1_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_X.xlsx','Sheet',1,'Range','B1');
writematrix(X2_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_X.xlsx','Sheet',1,'Range','C1');
writematrix(Y0_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_Y.xlsx','Sheet',1,'Range','A1');
writematrix(Y1_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_Y.xlsx','Sheet',1,'Range','B1');
writematrix(Y2_NB,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NB_Y.xlsx','Sheet',1,'Range','C1');

% Tree
writematrix(X0_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_X.xlsx','Sheet',1,'Range','A1');
writematrix(X1_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_X.xlsx','Sheet',1,'Range','B1');
writematrix(X2_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_X.xlsx','Sheet',1,'Range','C1');
writematrix(Y0_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_Y.xlsx','Sheet',1,'Range','A1');
writematrix(Y1_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_Y.xlsx','Sheet',1,'Range','B1');
writematrix(Y2_Tree,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_Tree_Y.xlsx','Sheet',1,'Range','C1');

% NN
writematrix(X0_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_X.xlsx','Sheet',1,'Range','A1');
writematrix(X1_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_X.xlsx','Sheet',1,'Range','B1');
writematrix(X2_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_X.xlsx','Sheet',1,'Range','C1');
writematrix(Y0_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_Y.xlsx','Sheet',1,'Range','A1');
writematrix(Y1_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_Y.xlsx','Sheet',1,'Range','B1');
writematrix(Y2_NN,'C:\Users\USER\Desktop\Meng-Ying\Output\ROC_NN_Y.xlsx','Sheet',1,'Range','C1');

%% AUC 
writematrix(AUC_0,'C:\Users\USER\Desktop\Meng-Ying\Output\AUC_0.xlsx');
writematrix(AUC_1,'C:\Users\USER\Desktop\Meng-Ying\Output\AUC_1.xlsx');
writematrix(AUC_2,'C:\Users\USER\Desktop\Meng-Ying\Output\AUC_2.xlsx');











