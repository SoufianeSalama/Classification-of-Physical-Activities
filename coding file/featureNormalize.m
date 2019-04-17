function [X_normalized] = featureNormalize(X)

X_normalized = X;
gemiddelde = zeros(1, size(X, 2));
standaardafw = zeros(1, size(X, 2));


for i = 1:size(X, 2)
    vector = X(:,i);
    gemiddelde(1,i) = mean(vector);
    standaardafw(1,i) = std(vector);
end

X_normalized = (X_normalized - gemiddelde) ./ standaardafw;

% ============================================================

end