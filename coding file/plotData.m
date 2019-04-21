function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


posactiviteit = find(y==1);
negactiviteit = find(y==0);

plot(X(negactiviteit, 1), X(negactiviteit, 2), 'rx', 'Markersize', 7)
plot(X(posactiviteit, 1), X(posactiviteit, 2), 'g+', 'Markersize', 7)

title('Ex1: Feature selection (Before Normalization)');
% xlabel('Feature 2: Max value of the acceleration at the x-axis of gravity acceleration');
% ylabel('Feature 6: Min value of the acceleration at the y-axis of gravity acceleration');

xlabel('Feature 1: Max value of the acceleration at the x-axis of body acceleration');
ylabel('Feature 8: Signal magnitude area of the 3-axis of gravity acceleration.');
hold off;

end
