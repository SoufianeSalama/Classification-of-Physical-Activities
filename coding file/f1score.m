function [f1score, precision, recall] = f1score(theta, X, y)

predictions = predict(theta, X);
confusionresult = confusionmat(y, predictions);

% figure; 
% confusionchart(y, predictions);
% title('Predicted vs True (Validation Set)');
%hold off;

% Foute berekening
% truepositive = confusionresult(2:2,2);
% falsepositive = confusionresult(2:2,1);
% falsenegative = confusionresult(1:1,1);

% Juiste Berekening
truepositive = confusionresult(2:2,2);
falsepositive = confusionresult(1:1,2);
falsenegative = confusionresult(2:2,1);

precision = truepositive/(truepositive+falsepositive);
recall = truepositive/(truepositive + falsenegative);

f1score = (2*precision*recall) / (precision + recall);

end
