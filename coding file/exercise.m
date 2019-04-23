%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression
%% Initialization
clear ; close all; clc
featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');

%% Exercise 1: Feature Selection
% Onze 1ste keuze
feature1 = 2; % Max value of the acceleration at the x-axis of gravity acceleration.
feature2 = 6; % Min value of the acceleration at the y-axis of gravity acceleration.
klasse = 6;


% Onze 2de keuze (te gebruiken als vergelijking tov keuze 1)
feature1 = 1; % Max value of the acceleration at the x-axis of body acceleration
feature2 = 8; % Signal magnitude area of the 3-axis of gravity acceleration.
klasse = 1;   % Walking

% Gegeven keuze
% feature1 = 4;
% feature2 = 6;
% klasse = 4;

%test 2.4
%klasse = 4;

% Normalizing the dataset
X_normalized = featureNormalize(featureData.features);
%X_normalized = featureData.features;

x1 = X_normalized(:, feature1);
x2 = X_normalized(:, feature2);
x0 = ones(size(x1(:,1)));

X = [x0,x1,x2];
Xallfeatures = [ones(size(x1(:,1))), X_normalized];
XallfeaturesTrainingGlobal = Xallfeatures(1:4120,:);
XallfeaturesValidationGlobal  = Xallfeatures(4121:7210,:);
XallfeaturesTestingGlobal  = Xallfeatures(7211:10299,:);


y = labelData.label(:, 1);
y1 = double(y==klasse);

plotData([x1,x2], y1);

% figure;hold on;
% gplotmatrix(featureData.features, [], y1);
% title("'Walking Up' Activity");
% hold off;

% Dataset delen in:
% ->40% training set
% ->30% (cross) validation set
% ->30% testing set

XtrainingGlobal = [x0(1:4120),x1(1:4120), x2(1:4120)];
ytraining = y1(1:4120, 1);

XvalidationGlobal = [x0(4121:7210), x1(4121:7210), x2(4121:7210)];
yvalidation = y1(4121:7210);

XtestingGlobal = [x0(7211:10299), x1(7211:10299), x2(7211:10299)];
ytesting = y1(7211:10299);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% Exercise 2: Classification: Logistic regression
% Implementing functions of assignment 2 and 5

%% 2.2 Linear model with 2 features

% Normalizing features is done in the beginning (mean normalization)

% Use training dataset to train the model (with lambda = 0)
lambda = 0;
initial_theta = zeros(size(XtrainingGlobal, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XtrainingGlobal , ytraining, lambda)), initial_theta, options);
fprintf('Ex2.2: Cost with FMINUNC (TRAINING DATASET) with lambda=0: %f\n', J);

% % Use validation dataset to test the model
% [cost, grad] = costFunctionReg(theta, XvalidationGlobal , yvalidation, lambda);
% fprintf('Ex2.2: Cost with FMINUNC (VALIDATION DATASET) with theta from TRAINING: %f\n', J);
% 
% % Plot the training dataset and the lineair decision boundery
% plotDecisionBoundary(theta, XtrainingGlobal , ytraining);
% title('Ex2.2: Training Set (lambda=0)');
% hold off;
% 
% % Plot the Cross Validation dataset and the lineair decision boundery
plotDecisionBoundary(theta, XvalidationGlobal , yvalidation);
title('Ex2.2: Cross Validation Set (lambda=0)');
hold off;

%Calculate F1 score -> https://www.coursera.org/learn/machine-learning/lecture/CuONQ/trading-off-precision-and-recall
% met behulp van confusionmat en https://en.wikipedia.org/wiki/F1_score
% F1Score of training set
f1scoreTraining = f1score(theta, XtrainingGlobal , ytraining);
fprintf('Ex2.2: F1 Score of the Training Set (Linear): %f\n', f1scoreTraining);
% F1Score of cross validation set
f1scoreValidation = f1score(theta, XvalidationGlobal , yvalidation);
fprintf('Ex2.2: F1 Score of the Validation Set (Linear): %f\n', f1scoreValidation);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% 2.3 Polynomial features from 2 features

% Map the features into a 28-dim vector (degree 6)
%X = mapFeature(X(:,1),X(:,2));
degree = 6;
Xtraining = XtrainingGlobal;
Xvalidation = XvalidationGlobal;

Xtraining(:,1) = [];
Xvalidation(:,1) = [];
Xtraining = mapFeature(Xtraining(:,1), Xtraining(:,2), degree);
Xvalidation = mapFeature(Xvalidation(:,1), Xvalidation(:,2), degree);
% 

%Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
herhalingen = 100;
lambda = logspace(-4.77, 4.77, herhalingen);
%lambda = linspace(3^(-10), 3^(10), herhalingen);
f1scoreMatrixTraining = [];
f1scoreMatrixCrossValidation=[];
initial_theta = zeros(size(Xtraining, 2), 1);
for i=1:herhalingen
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda(i) )), initial_theta, options);
    f1scoreMatrixTraining(:,end+1) = f1score(theta, Xtraining, ytraining);
    f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, Xvalidation, yvalidation);
end


figure; hold on;
plot(lambda, f1scoreMatrixTraining);
plot(lambda, f1scoreMatrixCrossValidation);
set(gca, 'XScale', 'log');
title('2.3: 2 features poly (6th degree)');
xlabel('Lambda');
ylabel('F1 Score');
legend('Training', 'Validation');
hold off;

lambda = 0.04863;
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda )), initial_theta, options);
    
% % % Plot the Cross Validation dataset and the lineair decision boundery
plotDecisionBoundary(theta, Xvalidation, yvalidation);
title('Ex2.3: Cross Validation Set (lambda=0.04863, F1Score=0.682199)');
hold off;

%Calculate F1 score -> https://www.coursera.org/learn/machine-learning/lecture/CuONQ/trading-off-precision-and-recall
% met behulp van confusionmat en https://en.wikipedia.org/wiki/F1_score
% F1Score of training set
% f1scoreTraining = f1score(theta, Xtraining, ytraining);
% fprintf('Ex2.3: F1 Score of the Training Set (Poly): %f\n', f1scoreTraining);

% F1Score of cross validation set
f1scoreValidation = f1score(theta, Xvalidation, yvalidation);
fprintf('Ex2.3: F1 Score of the Validation Set (Poly): %f\n', f1scoreValidation);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% 2.4.1 Construct a linear classifier with all eight features
% linear classifier -> de features niet mappen naar een bv 6de graad
XallfeaturesTrainingGlobal;
XallfeaturesValidationGlobal;
    
%Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
herhalingen = 200;
lambda = logspace(-4.77, 4.77, herhalingen);
%lambda = linspace(3^(-10), 3^(10), herhalingen);
f1scoreMatrixTraining = [];
f1scoreMatrixCrossValidation=[];
initial_theta = zeros(size(XallfeaturesTrainingGlobal, 2), 1);
for i=1:herhalingen
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTrainingGlobal, ytraining, lambda(i) )), initial_theta, options);
    f1scoreMatrixTraining(:,end+1) = f1score(theta, XallfeaturesTrainingGlobal, ytraining);
    f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, XallfeaturesValidationGlobal, yvalidation);
end
% 
% 
figure; hold on;
plot(lambda, f1scoreMatrixTraining);
plot(lambda, f1scoreMatrixCrossValidation);
set(gca, 'XScale', 'log');
title('2.4: 8 features linear');
xlabel('Lambda');
ylabel('F1 Score');
legend('Training', 'Validation');
xlim([0 1000])
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% 2.4.2 Construct a non-linear classifier with the 8 features
% linear classifier -> de features mappen naar bv 6de graad
XallfeaturesTraining = XallfeaturesTrainingGlobal;
XallfeaturesValidation = XallfeaturesValidationGlobal;

XallfeaturesTraining(:,1) = [];
XallfeaturesValidation(:,1) = [];
XallfeaturesTraining = mapFeatureMulti(XallfeaturesTraining,2);
XallfeaturesValidation = mapFeatureMulti(XallfeaturesValidation,2);

%Optimize Lambda (3^(-10) tot 3^(10)) with the validation dataset
herhalingen = 200;
lambda = logspace(-4.77, 4.77, herhalingen);
f1scoreMatrixTraining = [];
f1scoreMatrixCrossValidation=[];
initial_theta = zeros(size(XallfeaturesTraining, 2), 1);
for i=1:herhalingen
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTraining, ytraining, lambda(i) )), initial_theta, options);
    f1scoreMatrixTraining(:,end+1) = f1score(theta, XallfeaturesTraining, ytraining);
    f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, XallfeaturesValidation, yvalidation);
end
% 
% 
figure; hold on;
plot(lambda, f1scoreMatrixTraining);
plot(lambda, f1scoreMatrixCrossValidation);
set(gca, 'XScale', 'log');
title('2.4.2: 8 features non-linear (degree=2)');
xlabel('Lambda');
ylabel('F1 Score');
legend('Training', 'Validation');
hold off;

% F1 score tov aantal training voorbeelden
lambda = 0.01987; %lambda waarde die de hoogste F1 score geeft voor klasse 1
lambda = 78.27; %lambda waarde die de hoogste F1 score geeft voor klasse 4
lambda = 0.1;
options = optimset('GradObj', 'on', 'MaxIter', 100);
% [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTraining, ytraining, lambda )), initial_theta, options);
%   

f1scoreMatrixTraining = [];
f1scoreMatrixCrossValidation = [];
aantal = [];

[m, n] =size(Xvalidation);
i = 150 ;

while( i< m) 
    aantal(:,end+1) = i ;
   [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, XallfeaturesTraining(1:i, : ), ytraining(1:i, : ), lambda )), initial_theta, options);
  
    f1scoreMatrixTraining(:,end+1) = f1score(theta, XallfeaturesTraining(1:i, : ), ytraining(1:i, :));
    f1scoreMatrixCrossValidation(:,end+1) = f1score(theta, XallfeaturesValidation(1:i, : ), yvalidation(1:i, :));
    i = i +100 ;
end
figure; hold on;
plot(aantal, f1scoreMatrixTraining);
plot(aantal, f1scoreMatrixCrossValidation);
title('Adding more training examples');
xlabel('Number of training examples');
ylabel('F1 Score');
legend('Training','Validation');

%% Extra: Plotting Cost in function of the degree of polynomial for exercise 2.3 (VARIANCE VS BIAS)
i = 1 ;
max_degree = 6;
degrees = [];
error_train = [];
error_val = [];
lambda = 1;
initial_theta = zeros(size(Xtraining, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);

Xtraining = XtrainingGlobal;
Xvalidation = XvalidationGlobal;

Xtraining(:,1) = []; % x0 verwijderen
Xvalidation(:,1) = []; % x0 verwijderen
while(i<=max_degree) 
    degrees(:,end+1)=i ;
    Xtraining_temp = [];
    Xvalidation_temp = [];
    Xtraining_temp = mapFeature(Xtraining(:,1), Xtraining(:,2),i);
    Xvalidation_temp = mapFeature(Xvalidation(:,1), Xvalidation(:,2),i);
    initial_theta = zeros(size(Xtraining_temp, 2), 1);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining_temp, ytraining, lambda)), initial_theta, options);
    
    [error_train(:,end+1), grad] = costFunctionReg(theta, Xtraining_temp, ytraining, 0);
    [error_val(:,end+1), grad] = costFunctionReg(theta, Xvalidation_temp, yvalidation, 0);

    i = i+1 ;
end

figure; hold on;
plot(degrees, error_train);
plot(degrees, error_val);
title('Variance VS Bias (lambda=1)');
xlabel('Degree of polynomial');
ylabel('Error Cost(theta)');
legend('Training','Validation');

%% Extra: Plotting Cost in function of the degree of polynomial for exercise 2.4.2 (VARIANCE VS BIAS)
i = 1 ;
max_degree = 5;
degrees = [];
error_train = [];
error_val = [];
lambda = 1;
initial_theta = zeros(size(Xtraining, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);

XallfeaturesTraining = XallfeaturesTrainingGlobal;
XallfeaturesValidation = XallfeaturesValidationGlobal;

XallfeaturesTraining(:,1) = [];
XallfeaturesValidation(:,1) = [];

while(i<=max_degree) 
    degrees(:,end+1)=i ;
    Xtraining = mapFeatureMulti(XallfeaturesTraining,i);
    Xvalidation = mapFeatureMulti(XallfeaturesValidation,i);
    initial_theta = zeros(size(Xtraining, 2), 1);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xtraining, ytraining, lambda)), initial_theta, options);

    [error_train(:,end+1), grad] = costFunctionReg(theta, Xtraining, ytraining, 0);
    [error_val(:,end+1), grad] = costFunctionReg(theta, Xvalidation, yvalidation, 0);
    
    
    i = i+1;
end

figure; hold on;
plot(degrees, error_train);
plot(degrees, error_val);
title('Variance VS Bias Poly 8 Features (lambda=1)');
xlabel('Degree of polynomial');
ylabel('Error Cost(theta)');
legend('Training','Validation');
