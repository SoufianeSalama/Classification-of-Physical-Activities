function result = mg_sums(n_numbers, d)
if(n_numbers<=1)
    result = d;
else
    result = zeros(0, n_numbers);    
    for(i = d:-1:0)
        rc = mg_sums(n_numbers - 1, d - i);
        result = [result; i * ones(size(rc,1), 1), rc];
    end    
end