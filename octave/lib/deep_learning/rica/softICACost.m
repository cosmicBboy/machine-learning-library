%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

% Do forward prop first to get reconstruction term
z2 = W * x;
reconstructionError = W' * z2 - x;

% Compute cost with regularization term
l1_norm = sqrt(z2 .^ 2 + params.epsilon);
rg_term = params.lambda * sum(sum(l1_norm));
rcn_term = 0.5 * sum(sum((reconstructionError .^ 2)));
cost = rg_term + rcn_term;

% Compute gradient Wgrad
% The reconstructionError should be multiplied by 2, but since the
% cost term is scaled back by 0.5, we similarly scale back
% the reconstructionErrors here.
Wgrad = (W * (reconstructionError) * x') + ...
        z2 * (reconstructionError)';

% NOTE in the notation in
% http://ufldl.stanford.edu/tutorial/unsupervised/ExerciseRICA/
% the cost term scales back the reconstruction term by 0.5, which is why
% we don't multiply the reconstruction term by 2 in the Wgrad
% computation

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
