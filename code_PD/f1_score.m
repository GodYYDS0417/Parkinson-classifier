function [f1_score] = f1_score(confusionmat)

C = confusionmat';
diagonal = diag(C);
sum_of_rows = sum(C,2);
sum_of_cols = sum(C,1);

percision =  diagonal ./ sum_of_rows ;
recall =  diagonal ./ sum_of_cols' ;
for i=1:3
    f1_score(i) = 2*(percision(i)*recall(i))/(percision(i)+recall(i));
end