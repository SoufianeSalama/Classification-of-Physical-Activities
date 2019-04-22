function plotDecisionBoundary(theta, X, y)
%function plotDecisionBoundary(theta, X1, X2, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data

plotData(X(:,2:3), y);

hold on;

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Negative-train','Positive-train', 'Decision boundary');
    %axis([-3,3,-2,2])
    axis([-6,6,-4,4])
else
    % Here is the grid range
%     u = linspace(-1, 1.5, 50);
%     v = linspace(-1, 1.5, 50);
    
    u = linspace(-3, 3, 50);
    v = linspace(-3, 3, 50);

    z = zeros(length(u), length(v));
    degree = 6;
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j), degree)*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
    axis([-2,3,-3,4])
%  end
% Put some labels 
hold on;
% Labels and Legend

% Specified in plot order
legend('Negative-train','Positive-train', 'Decision boundary');
hold off;

end
