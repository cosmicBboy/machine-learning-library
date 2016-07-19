theta =[1; 2; 1] % thetas as column vector
x_i =[1; 1; 3] % a single training example as column vector

% unvectorized implementation
n = length(theta)
prediction = 0.0;
for j = 1:n,
    prediction = prediction + theta(j) * x_i(j);
end;
disp('Unvectorized:')
prediction

% vectorized implementation
disp('Vectorized:')
theta' * x_i

% example: normalizing many vectors
X = [1 2; 3 4; 5 6]
% compute the row-wise sum of squares
X_norm = sqrt(sum(X .^2, 1 ));
% divide each row with X_norm
Y = bsxfun(@rdivide, X, X_norm);
