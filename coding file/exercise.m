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
y1 = double(y==6);

figure; hold on;

%gplotmatrix(featureData.features, [], y1);
%hold off;
%pause;
% 
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7)
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7)
% 
% 
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

% Dataset delen in:
% ->40% training set
% ->30% validation set
% ->30% testing set

Xtraining = [x1(1:4120), x2(1:4120)];
ytraining = y1(1:4120, 1);

Xvalidation = [x1(4121:7210), x2(4121:7210)];
yvalidation = y1(4121:7210);

Xtesting = [x1(7211:10299), x2(7211:10299)];
ytesting = y1(7211:10299);

%% Exercise 2: Classification: Logistic regression
%% 2.1 Cost function and gradient

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

%X = mapFeature(X(:,1),X(:,2));
Xtraining = mapFeature(Xtraining(:,1), Xtraining(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(Xtraining, 2), 1);

% Set regularization parameter lambda to 1
lambda = 10;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, Xtraining, ytraining, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda)), initial_theta, options);
fprintf('Cost with FMINUNC: %f\n', J);

J

plotDecisionBoundary(theta, Xtraining, ytraining);

% Compute accuracy on our training set
p = predict(theta, Xtraining);

fprintf('Train Accuracy: %f\n', mean(double(p == ytraining)) * 100);
