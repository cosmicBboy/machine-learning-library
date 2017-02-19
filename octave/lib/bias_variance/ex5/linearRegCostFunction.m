function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
zeroed_theta = [0; theta(2:end)];

% Compute cost
h_x = X * theta;
sqrs = (h_x - y) .^ 2;
sum_sqrs = sum(sqrs);
reg_term = (lambda / (2 * m)) * sum(zeroed_theta .^ 2);
J = ((1 / (2 * m)) * sum_sqrs) + reg_term;

% Compute gradient
grad_reg = zeroed_theta .* (lambda / m);
grad = ((h_x - y)' * X) ./ m + grad_reg';


% =========================================================================

grad = grad(:);

end
