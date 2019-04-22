function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE ======================
% You need to finish the cost function and calculate the gradient result in
% here.

h = sigmoid(X*theta);

J = (-1/m)* (y'*log(h) + (1-y)' *log(1-h)) + (lambda/(2*m)) * (theta' * theta);
%J = (1/(2*m)) * sum((h - y).^2) + (lambda/(2*m)) * (theta' * theta);
%J = (-1/m)* (y'*log(h) + (1-y)' *log(1-h));

thetaNul = theta;
thetaNul(1) = 0;

grad = ((1/m) * (h-y)' * X) + (lambda/m)*thetaNul';


% =============================================================

end
