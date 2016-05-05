function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Feedforward for layer 2 (hidden layer)
% --------------------------------------
% A2 should result in an m * a_2 matrix where:
% - m is the number of examples
% - a_2 is the number of nodes in the 2nd layer
Z2 = X * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];

disp('Dimensions of A2')
disp(size(A2))

% Feedforward for layer 3 (output layer)
% --------------------------------------
% A3 should result in an m * c matrix where:
% - m is the number of examples
% - c is the number of classes to be predicted
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

disp('Dimensions of A3')
disp(size(A3))

[values, p] = max(A3, [], 2);

% p = round(a3);
% =========================================================================


end
