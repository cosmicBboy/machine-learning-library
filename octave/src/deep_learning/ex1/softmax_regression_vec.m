function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %

  % first compute the hypothesis. This should be an k-by-m matrix, where m are
  % the number of examples and k are the number of classes.
  % this is a k-by-m matrix
  % IMPORTANT: here we want to add a column of exp(0) = 1's, since we assume
  % that the thetas for theta(:,num_classes) = 0.
  h_prob = [exp(theta' * X); ones(1, m)];
  % use bsxfun to normalize the predicted probabilities for each example
  h_x = bsxfun(@rdivide, h_prob, sum(h_prob, 1));

  % create a k-by-m boolean matrix where for each column (example), the value
  % should be 0 except for the row index where the index equals the true label
  % for that example.
  indicator_matrix = repmat((1:(num_classes))', 1, m) == y;

  % then compute the cost by multiplying the indicator matrix with h_x, then
  % summing over each class and each example
  f = -sum(sum(indicator_matrix .* log(h_x)));

  % finally, compute the gradient of the parameters for each class. This should
  % by a n-by-k matrix, where n are the number of features in the data.
  % NOTE: We want to select only (num_classes - 1) columns so that we fulfill
  % our assumption that theta(:,num_classes) = 0
  g = -(X * (indicator_matrix .- h_x)')(:, 1:(num_classes - 1));
  g = g(:); % make gradient a vector for minFunc
