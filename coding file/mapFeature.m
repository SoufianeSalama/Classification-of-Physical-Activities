%function out = mapFeature(X, pair, degree)
function out = mapFeature(X1,X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to features used in the regularization exercise, which has the
%   same function of x2fx().
%   pair is the optimal feature pairs selected after obervation
%   degree is the degree of the polynomial features, not only quadratic


degree = 6;
% size(X);
% X1 = X(:,1);
% X2 = X(:,2);

out = ones(size(X1(:,1))); %-> word al in het begin van 'exercise' gedaan

for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end



end