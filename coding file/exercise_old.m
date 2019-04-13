%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression
%% Initialization
clear ; close all; clc
%% Exercise 1: Feature Selection

featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');

x1 = featureData.features(1:100, 4);
x2 = featureData.features(1:100, 6);

X = [x1,x2];
xmiche1 = X;

y = labelData.label(1:100, 1);
y1 = double(y==6);

figure; hold on;

posactiviteit = find(y==6);
negactiviteit = find(y~=6);

%gplotmatrix(featureData.features, [], y1);
hold off;
pause;
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


%% Exercise 2: Classification: Logistic regression
%% 2.1 Cost function and gradient

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

X = mapFeature(X(:,1),X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 10;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);
X
% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
fprintf('Cost with FMINUNC: %f\n', J);

J
Xmichel2 = X
plotDecisionBoundary(theta, X, y);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
