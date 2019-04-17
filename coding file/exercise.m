%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression
%% Initialization
clear ; close all; clc
featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');

%% Exercise 1: Feature Selection
% Onze keuze
feature1 = 2;
feature2 = 6;
klasse = 6;

% Gegeven keuze
feature1 = 4;
feature2 = 6;
klasse = 4;

% Normalizing the dataset
X_normalized = featureNormalize(featureData.features);
%X_normalized = featureData.features;

x1 = X_normalized(:, feature1);
x2 = X_normalized(:, feature2);
x0 = ones(size(x1(:,1)));

X = [x0,x1,x2];
Xallfeatures = [ones(size(x1(:,1))), X_normalized];
XallfeaturesTraining = Xallfeatures(1:4120,:);
XallfeaturesValidation = Xallfeatures(4121:7210,:);
XallfeaturesTesting = Xallfeatures(7211:10299,:);


y = labelData.label(:, 1);
y1 = double(y==klasse);

plotData([x1,x2], y1);

figure;hold on;
gplotmatrix(featureData.features, [], y1);
title("'Standing' Activity");
hold off;

% Dataset delen in:
% ->40% training set
% ->30% (cross) validation set
% ->30% testing set

Xtraining = [x0(1:4120),x1(1:4120), x2(1:4120)];
ytraining = y1(1:4120, 1);

Xvalidation = [x0(4121:7210), x1(4121:7210), x2(4121:7210)];
yvalidation = y1(4121:7210);

Xtesting = [x0(7211:10299), x1(7211:10299), x2(7211:10299)];
ytesting = y1(7211:10299);
% 
% %% Exercise 2: Classification: Logistic regression
% %% 2.1 Cost function and gradient
% 
% %  Setup the data matrix appropriately, and add ones for the intercept term
% [m, n] = size(X);
% 
% %X = mapFeature(X(:,1),X(:,2));
% %Xtraining = mapFeature(Xtraining(:,1), Xtraining(:,2));
% %Xvalidation = mapFeature(Xvalidation(:,1), Xvalidation(:,2));
% % Initialize fitting parameters
% initial_theta = zeros(size(Xtraining, 2), 1);
% 
% lambda = 1;
% % Compute and display initial cost and gradient for regularized logistic regression
% [cost, grad] = costFunctionReg(initial_theta, Xtraining, ytraining, lambda);
% fprintf('Cost at initial theta (zeros): %f\n', cost);
% 
% 
%% 2.2 Linear model with 2 features

% Normalizing features is done in the beginning (mean normalization)

% Use training dataset to train the model (with lambda = 0)
lambda = 0;
initial_theta = zeros(size(Xtraining, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda)), initial_theta, options);
fprintf('Ex2.2: Cost with FMINUNC (TRAINING DATASET) with lambda=0: %f\n', J);

% Use validation dataset to test the model
[cost, grad] = costFunctionReg(theta, Xvalidation, yvalidation, lambda);
fprintf('Ex2.2: Cost with FMINUNC (VALIDATION DATASET) with theta from TRAINING: %f\n', J);

% Plot the training dataset and the lineair decision boundery
plotDecisionBoundary(theta, Xtraining, ytraining);
title('Ex2.2: Training Set (lambda=0)');
hold off;

% % % Plot the Cross Validation dataset and the lineair decision boundery
plotDecisionBoundary(theta, Xvalidation, yvalidation);
title('Ex2.2: Cross Validation Set (lambda=0)');
hold off;

%Calculate F1 score -> https://www.coursera.org/learn/machine-learning/lecture/CuONQ/trading-off-precision-and-recall
% met behulp van confusionmat en https://en.wikipedia.org/wiki/F1_score
% F1Score of training set
f1scoreTraining = f1score(theta, Xtraining, ytraining);
fprintf('Ex2.2: F1 Score of the Training Set: %f\n', f1scoreTraining);
% F1Score of cross validation set
f1scoreValidation = f1score(theta, Xvalidation, yvalidation);
fprintf('Ex2.2: F1 Score of the Validation Set: %f\n', f1scoreValidation);

% 
% 
% % Compute accuracy on our training set
p = predict(theta, Xtraining);
fprintf('Train Accuracy: %f\n', mean(double(p == ytraining)) * 100);
% 
%% 2.3 Polynomial features from 2 features

% Map the features into a 28-dim vector (degree 6)
%X = mapFeature(X(:,1),X(:,2));
%degree = 6;
Xtraining(:,1) = [];
Xvalidation(:,1) = [];
Xtraining = mapFeature(Xtraining(:,1), Xtraining(:,2));
Xvalidation = mapFeature(Xvalidation(:,1), Xvalidation(:,2));


%Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
% herhalingen = 200;
% lambda = logspace(-10, 10, herhalingen);
% %lambda = linspace(3^(-10), 3^(10), herhalingen);
% f1scoreMatrixTraining = [];
% f1scoreMatrixCrossValidation=[];
 initial_theta = zeros(size(Xtraining, 2), 1);
% for i=1:herhalingen
%     options = optimset('GradObj', 'on', 'MaxIter', 200);
%     [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda(i) )), initial_theta, options);
%     f1scoreMatrixTraining(:,end+1) = f1score(theta, Xtraining, ytraining);
%     f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, Xvalidation, yvalidation);
% end
% % 
% % 
% figure; hold on;
% plot(lambda, f1scoreMatrixTraining);
% plot(lambda, f1scoreMatrixCrossValidation);
% set(gca, 'XScale', 'log');
% title('2.3: 2 features poly (6th degree)');
% xlabel('Lambda');
% ylabel('F1 Score');
% legend('Training', 'Validation');
% hold off;

lambda = 0;
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda )), initial_theta, options);
    
% Plot the Cross Validation dataset and the lineair decision boundery
plotDecisionBoundary(theta, Xtraining, ytraining);
%plotDecisionBoundary(theta, Xvalidation(:,2:2), Xvalidation(:,3:3), yvalidation);
title('Ex2.3: Cross Validation Set (lambda=0, F1Score=...)');
xlabel('Feature 2');
ylabel('Feature 4');
hold off;
% 

%% 2.4 Construct a linear classifier with all eight features
% linear classifier -> de features niet mappen naar een bv 6de graad
% XallfeaturesTraining;
% XallfeaturesValidation;
%     
% %Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
% herhalingen = 200;
% lambda = logspace(-10, 10, herhalingen);
% %lambda = linspace(3^(-10), 3^(10), herhalingen);
% f1scoreMatrixTraining = [];
% f1scoreMatrixCrossValidation=[];
% initial_theta = zeros(size(XallfeaturesTraining, 2), 1);
% for i=1:herhalingen
%     options = optimset('GradObj', 'on', 'MaxIter', 100);
%     [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTraining, ytraining, lambda(i) )), initial_theta, options);
%     f1scoreMatrixTraining(:,end+1) = f1score(theta, XallfeaturesTraining, ytraining);
%     f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, XallfeaturesValidation, yvalidation);
% end
% % 
% % 
% figure; hold on;
% plot(lambda, f1scoreMatrixTraining);
% plot(lambda, f1scoreMatrixCrossValidation);
% set(gca, 'XScale', 'log');
% title('2.4: 8 features linear ');
% xlabel('Lambda');ans
% ylabel('F1 Score');
% legend('Training', 'Validation');
% hold off;


%% 2.4 Construct a non-linear classifier with the 8 features
% linear classifier -> de features mappen naar bv 6de graad
% XallfeaturesTraining(:,1) = [];
% XallfeaturesValidation(:,1) = [];
% XallfeaturesTraining = mapFeatureMulti(XallfeaturesTraining);
% XallfeaturesValidation = mapFeatureMulti(XallfeaturesValidation);
% 
% %Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
% herhalingen = 200;
% lambda = logspace(-10, 10, herhalingen);
% %lambda = linspace(3^(-10), 3^(10), herhalingen);
% f1scoreMatrixTraining = [];
% f1scoreMatrixCrossValidation=[];
% initial_theta = zeros(size(XallfeaturesTraining, 2), 1);
% for i=1:herhalingen
%     options = optimset('GradObj', 'on', 'MaxIter', 100);
%     [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTraining, ytraining, lambda(i) )), initial_theta, options);
%     f1scoreMatrixTraining(:,end+1) = f1score(theta, XallfeaturesTraining, ytraining);
%     f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, XallfeaturesValidation, yvalidation);
% end
% % 
% % 
% figure; hold on;
% plot(lambda, f1scoreMatrixTraining);
% plot(lambda, f1scoreMatrixCrossValidation);
% set(gca, 'XScale', 'log');
% title('2.4: 8 features non-linear (degree=2)');
% xlabel('Lambda');
% ylabel('F1 Score');
% legend('Training', 'Validation');
% hold off;
% 
% size(XallfeaturesTraining(1:200,1))
% 
% figure; hold on;
% plot(f1scoreMatrixTraining);
% %plot(size(XallfeaturesValidation(:,1)), f1scoreMatrixCrossValidation);
% title('2.4: Adding more trainings examples');
% xlabel('Number of training examples');
% ylabel('F1 Score');
% legend('Training', 'Validation');
% hold off;