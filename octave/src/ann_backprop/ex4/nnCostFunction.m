function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% (input_layer_size + 1) => input layer features + bias unit
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Feedforward for hidden layer
Z2 = X * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];

% Feedforward for output layer
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% Computing the Cost Function

% Create a m x k matrix for values of y, where each row is a vector of 0's
% except for the kth column representing the label, for example:
% The 5 label would be the row vector [0 0 0 0 1 0 0 0 0 0]
% This allows us to vectorize the cost function implementaion
Y = repmat(1:num_labels, size(y)) == repmat(y, 1, num_labels);

% We want to select from the 2nd onward columnd of Theta1 and Theta2...
% this is because first column in the Theta1 and Theta2 matrices correspond to
% the weights on the bias nodes for the input and hidden layers respectively
Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);
theta_reg_sum = sum(sum(Theta1_no_bias .^ 2)) + sum(sum(Theta2_no_bias .^ 2));
reg_term = (lambda / (2 * m)) * theta_reg_sum;

J = (1 / m) * sum(sum(-(Y) .* log(A3) .- (1 - Y) .* log(1 - A3))) + reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% Vectorized Implementation of Backpropagation
D_3 = A3 .- Y;
D_2 = (D_3 * Theta2_no_bias) .* sigmoidGradient(Z2);
Grad_2 = D_3' * A2;
Grad_1 = D_2' * X;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_reg = [zeros(hidden_layer_size, 1) (lambda / m) * Theta1(:, 2:end)];
Theta2_reg = [zeros(num_labels, 1) (lambda / m) * Theta2(:, 2:end)];

Theta1_grad = (1 / m) * Grad_1 .+ Theta1_reg;
Theta2_grad = (1 / m) * Grad_2 .+ Theta2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
