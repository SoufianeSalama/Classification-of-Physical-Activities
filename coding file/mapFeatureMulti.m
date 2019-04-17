function out = mapFeatureMulti(X)
% X
% degree = 2;
% [m,n] = size(X);
% n
% %X(:,1) = [];
% out = ones(size(X(:,1))); %-> word al in het begin van 'exercise' gedaan
% b=0;
% test=1;
% for i = 1:n
%     out(:, end+1) = X(:,i).^2;
%     if i>1
%         for j = 1:test
%             out(:, end+1) = X(:,j)*X(:,i);
%             test = test + 1;
%         end
%     end
% 
% end


degree = 2;
[m,n] = size(X);

stacked = zeros(0, n); %this will collect all the coefficients...    
for(d = 1:degree)          % for degree 1 polynomial to degree 'order'
    stacked = [stacked; mg_sums(n, d)];
end

for(i = 1:size(stacked,1))
    accumulator = ones(m, 1);
    for(j = 1:n)
        accumulator = accumulator .* X(:,j).^stacked(i,j);
    end
    newX(:,i) = accumulator;
end

out = [ones(m,1),newX];

end
%% met hulp van https://stackoverflow.com/questions/33660799/feature-mapping-using-multi-variable-polynomial
