function [f1score, precision, recall] = f1score(theta, X, y)

predictions = predict(theta, X);
confusionresult = confusionmat(y, predictions);
%confusionchart(y, predictions);

truepositive = confusionresult(2:2,2);
%actualpositive = size(ytraining==1);
falsepositive = confusionresult(2:2,1);
falsenegative = confusionresult(1:1,1);

precision = truepositive/(truepositive+falsepositive);
recall = truepositive/(truepositive + falsenegative);

f1score = (2*precision*recall) / (precision + recall);

end
