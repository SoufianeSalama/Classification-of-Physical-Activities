%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression
clear ; close all; clc

featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');
y = labelData.label(:, 1);
posactiviteit = find(y==4);
negactiviteit = find(y~=4);




%% Feature 1 and Feature 2
% x1 = featureData.features(:, 1);
% x2 = featureData.features(:, 2);
% 
% X = [x1,x2];
% 
% figure; hold on;
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% 
% xlabel('Feature 1 (Max value of the acceleration at the x-axis of body acceleration)');
% ylabel('Feature 2 (Max value of the acceleration at the x-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 2 and Feature 3
% x1 = featureData.features(:, 2);
% x2 = featureData.features(:, 3);
% 
% X = [x1,x2];
% 
% figure; hold on;
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% 
% xlabel('Feature 2 (Max value of the acceleration at the x-axis of gravity acceleration)');
% ylabel('Feature 3 (Max value of the acceleration at the y-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 3 and Feature 4
% x1 = featureData.features(:, 3);
% x2 = featureData.features(:, 4);
% 
% X = [x1,x2];
% 
% figure; hold on;
% 
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% 
% xlabel('Feature 3 (Max value of the acceleration at the y-axis of gravity acceleration)');
% ylabel('Feature 4 (Max value of the acceleration at the z-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 4 and Feature 5
% x1 = featureData.features(:, 4);
% x2 = featureData.features(:, 5);
% 
% X = [x1,x2];
% 
% figure; hold on;
% 
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% 
% xlabel('Feature 4 (Max value of the acceleration at the z-axis of gravity acceleration)');
% ylabel('Feature 5 (Min value of the acceleration at the x-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 5 and Feature 6
% x1 = featureData.features(:, 5);
% x2 = featureData.features(:, 6);
% 
% X = [x1,x2];
% 
% figure; hold on;
% 
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% 
% xlabel('Feature 5 (Min value of the acceleration at the x-axis of gravity acceleration)');
% ylabel('Feature 6 (Min value of the acceleration at the y-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 6 and Feature 7
% x1 = featureData.features(:, 6);
% x2 = featureData.features(:, 7);
% 
% X = [x1,x2];
% 
% figure; hold on;
% 
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% 
% xlabel('Feature 6 (Min value of the acceleration at the y-axis of gravity acceleration)');
% ylabel('Feature 7 (Min value of the acceleration at the z-axis of gravity acceleration)')
% 
% Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;


%% Feature 5 and Feature 8
% x1 = featureData.features(:, 5);
% x2 = featureData.features(:, 8);
% 
% X = [x1,x2];
% 
% figure; hold on;
% plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
% plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);
% 
% xlabel('Feature 5 (Min value of the acceleration at the x-axis of gravity acceleration)');
% ylabel('Feature 8 (Signal magnitude area of the 3-axis of gravity acceleration)')
% 
% % Specified in plot order
% legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
% hold off;



%% Feature 4 6 en 7

x1 = featureData.features(:, 4);
x2 = featureData.features(:, 6);
x3 = featureData.features(:, 7);

X = [x1,x2,x3];

figure; hold on;
plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7);
plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7);

plot(X(:, 2), X(:, 3), 'bo', 'Markersize', 3);


xlabel('Feature 5 (Min value of the acceleration at the x-axis of gravity acceleration)');
ylabel('Feature 8 (Signal magnitude area of the 3-axis of gravity acceleration)')

% Specified in plot order
legend('y=1 (Wandelen naar boven)', 'y=0(NIET Wandelen naar boven)')
hold off;

