%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression
%% Initialization
clear ; close all; clc
%% Exercise 1: Feature Selection

featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');
x1 = featureData.features(:, 2);
x2 = featureData.features(:, 6);
X = [x1,x2];
y = labelData.label(:, 1);
y1=double(y==6);

plotData(X ,y1);

figure; hold on;

%splitting the dataset 
[m,n] = size(X) ;

x1training = featureData.features(1:412, 4);
x2training = featureData.features(1:412, 6);
ytraining = double(labelData.label(1:412, 1)==6) ;
ytemp = labelData.label(1:412, 1);
xtraining = [x1training, x2training ];
plotData(xtraining, ytraining) ;

x1testing = featureData.features(413:10299, 4);
x2testing = featureData.features(413:10299, 6);

%%random opdeling maar dan weet je de klassen wel niet 
% idx = randperm(m);
% Training = X(idx(1:round(P*m)),:);
% Testing = X(idx(round(P*m)+1:end),:);

%hier plot je nu eigenlijk een feature tegen de rest van de features.
%(wordt bepaaldt door y1)

 
 posactiviteit = find(y==4);
% negactiviteit = find(y~=4);
 % gplotmatrix(featureData.features , [], y1);

% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7)
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7)
% 
% 
% hold off;
% % Put some labels
% %hold on;
% % Labels and Legend
% xlabel('Feature 6 (Min value of the acceleration at the y-axis of gravity acceleration)')
% ylabel('Feature 7 (Min value of the acceleration at the z-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('Y=0', 'Y=1')
% hold off;


%% Exercise 2: Classification: Logistic regression
%% 2.1 Cost function and gradient
%2.1.1 Accomplish the sigmoid function 

%2.1.2 finish the cost function and gradient function. (Regularized)

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(xtraining);
xtrainingold = xtraining ;
disp(xtraining);
% Add intercept term to x and X_test 
xtraining = [ones(m, 1) xtraining];
xtraining = mapFeature(xtraining(:,2),xtraining(:,3));
disp(xtraining);
% Initialize fitting parameters
initial_theta = zeros(size(xtraining, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, xtraining, ytraining, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, xtraining, ytraining, lambda)), initial_theta, options);

%size(theta)

%size(X)

%J
disp(xtraining);
disp(xtraining(:, 1:2));
plotDecisionBoundary(theta, xtraining, ytraining);
hold on; 
title(sprintf('lambda = %g', lambda))

hold off ; 