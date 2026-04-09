% plot all figure %

g1 = repmat({'Tree'},epoch,1);
g2 = repmat({'SVM'},epoch,1);
g3 = repmat({'KNN'},epoch,1);
g4 = repmat({'LD'},epoch,1);
g5 = repmat({'NB'},epoch,1);
g6 = repmat({'NN'},epoch,1);
g= [g1;g2;g3;g4;g5;g6];

%% boxplot accuracy
figure(1);
boxplot(Accuracy',g')
ylabel('Testing accuracy')
% title('Accuracy')

%% boxplot f1score
figure(2);
boxplot(F1_score_0,g)
ylabel('F1 score')
title('F1 score for class "0"')

figure(3);
boxplot(F1_score_1,g)
ylabel('F1 score')
title('F1 score for class "1"')

figure(4);
boxplot(F1_score_2,g)
ylabel('F1 score')
title('F1 score for  class "2"')

%% boxplot AUC
figure(5);
boxplot(AUC_0,g)
ylabel('AUC')
title('AUC for class "0"')

figure(6);
boxplot(AUC_1,g)
ylabel('AUC')
title('AUC for class "1"')

figure(7);
boxplot(AUC_2,g)
ylabel('AUC')
title('AUC for class "2"')

%% plot ROC curve
maker_idx=round(linspace(1,50,50/5));
% class 0
figure(8);
h1=plot(X0_KNN,Y0_KNN,'-diamond','color',[255,48,48]/255,'linewidth',2,'markersize',8);
set(h1, 'MarkerFaceColor', get (h1, 'color' ));
hold on;

h2=plot(X0_LD,Y0_LD,'--s','color',[107,142,35]/255,'linewidth',2,'markersize',8);
set(h2, 'MarkerFaceColor', get (h2, 'color' ));
hold on;

h3=plot(X0_NB,Y0_NB,'-.o', 'color',[0,0,205]/255,'linewidth',2,'markersize',8);
set(h3, 'MarkerFaceColor', get (h3, 'color' ));
hold on;

h4=plot(X0_SVM,Y0_SVM,'-->', 'MarkerIndices',maker_idx, 'color',[242,159,5]/255,'linewidth',2,'markersize',8);
set(h4, 'MarkerFaceColor', get (h4, 'color' ));
hold on;

h5=plot(X0_Tree,Y0_Tree,':v',  'color',[62,96,111]/255,'linewidth',2,'markersize',8);
set(h5, 'MarkerFaceColor', get (h5, 'color' ));
hold on;

h6=plot(X0_NN,Y0_NN,':p', 'MarkerIndices',maker_idx, 'color',[30,30,32]/255,'linewidth',2,'markersize',8);
set(h6, 'MarkerFaceColor', get (h6, 'color' ));

title('Class:0');
h=legend('KNN','LD','NB','SVM','Tree','NN');
set(h,'Interpreter','latex','FontSize', 15)
xlabel('False positive rate')
ylabel('True positive rate')
grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'ytick',[0.0, 0.2,0.4,0.6, 0.8,1.0]);
ylim([0.0,1.0]);
set(gca,'xtick',[0.0, 0.2,0.4,0.6, 0.8,1.0]);
set(gca,'gridcolor',[0.8,0.8,0.8],'gridlinestyle',':','gridalpha',1,'linewidth',1)
box on;

% class 1
figure(9);
h1=plot(X1_KNN,Y1_KNN,'-diamond','color',[255,48,48]/255,'linewidth',2,'markersize',8);
set(h1, 'MarkerFaceColor', get (h1, 'color' ));
hold on;

h2=plot(X1_LD,Y1_LD,'--s','color',[107,142,35]/255,'linewidth',2,'markersize',8);
set(h2, 'MarkerFaceColor', get (h2, 'color' ));
hold on;

h3=plot(X1_NB,Y1_NB,'-.o',  'color',[0,0,205]/255,'linewidth',2,'markersize',8);
set(h3, 'MarkerFaceColor', get (h3, 'color' ));
hold on;

h4=plot(X1_SVM,Y1_SVM,'-->', 'MarkerIndices',maker_idx, 'color',[242,159,5]/255,'linewidth',2,'markersize',8);
set(h4, 'MarkerFaceColor', get (h4, 'color' ));
hold on;

h5=plot(X1_Tree,Y1_Tree,':v', 'color',[62,96,111]/255,'linewidth',2,'markersize',8);
set(h5, 'MarkerFaceColor', get (h5, 'color' ));
hold on;

h6=plot(X1_NN,Y1_NN,':p', 'MarkerIndices',maker_idx, 'color',[30,30,32]/255,'linewidth',2,'markersize',8);
set(h6, 'MarkerFaceColor', get (h6, 'color' ));

title('Class:1');
h=legend('KNN','LD','NB','SVM','Tree','NN');
set(h,'Interpreter','latex','FontSize', 15)
xlabel('False positive rate')
ylabel('True positive rate')
grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'ytick',[0.0, 0.2,0.4,0.6, 0.8,1.0]);
ylim([0.0,1.0]);
set(gca,'xtick',[0.0, 0.2,0.4,0.6, 0.8,1.0]);
set(gca,'gridcolor',[0.8,0.8,0.8],'gridlinestyle',':','gridalpha',1,'linewidth',1)
box on;

% class 2
figure(10);
h1=plot(X2_KNN,Y2_KNN,'-diamond','color',[255,48,48]/255,'linewidth',2,'markersize',8);
set(h1, 'MarkerFaceColor', get (h1, 'color' ));
hold on;

h2=plot(X2_LD,Y2_LD,'--s','color',[107,142,35]/255,'linewidth',2,'markersize',8);
set(h2, 'MarkerFaceColor', get (h2, 'color' ));
hold on;

h3=plot(X2_NB,Y2_NB,'-.o',  'color',[0,0,205]/255,'linewidth',2,'markersize',8);
set(h3, 'MarkerFaceColor', get (h3, 'color' ));
hold on;

h4=plot(X2_SVM,Y2_SVM,'-->', 'MarkerIndices',maker_idx, 'color',[242,159,5]/255,'linewidth',2,'markersize',8);
set(h4, 'MarkerFaceColor', get (h4, 'color' ));
hold on;

h5=plot(X2_Tree,Y2_Tree,':v', 'color',[62,96,111]/255,'linewidth',2,'markersize',8);
set(h5, 'MarkerFaceColor', get (h5, 'color' ));
hold on;

h6=plot(X2_NN,Y2_NN,':p', 'MarkerIndices',maker_idx, 'color',[30,30,32]/255,'linewidth',2,'markersize',8);
set(h6, 'MarkerFaceColor', get (h6, 'color' ));
title('Class:2');
h=legend('KNN','LD','NB','SVM','Tree','NN');
set(h,'Interpreter','latex','FontSize', 15)
xlabel('False positive rate')
ylabel('True positive rate')
grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'ytick',[0.0, 0.2,0.4,0.6, 0.8,1.0]);
ylim([0.0,1.0]);
set(gca,'xtick',[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
xlim([0.0,1.0]);
set(gca,'gridcolor',[0.8,0.8,0.8],'gridlinestyle',':','gridalpha',1,'linewidth',1)
box on;




