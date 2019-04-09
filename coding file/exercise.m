%% Machine Learning Lab Assignment: Classification of physical activities with Logistic Regression


%% Exercise 1: Feature Selection

featureData = load('..\Dataset\Features.mat');
labelData = load('..\Dataset\Label.mat');

x1 = featureData.features(:, 1);
x2 = featureData.features(:, 8);

X = [x1,x2];
y = labelData.label(:, 1);

%plot(x1,x2, 'rx');

figure; hold on;

posactiviteit = find(y==2);
negactiviteit = find(y~=2);

plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7)
plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7)


hold off;
% Put some labels 
%hold on;
% Labels and Legend
xlabel('Feature 2')
ylabel('Feature 1')

% Specified in plot order
legend('Y=1', 'Y=0')
hold off;

