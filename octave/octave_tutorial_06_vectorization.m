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
