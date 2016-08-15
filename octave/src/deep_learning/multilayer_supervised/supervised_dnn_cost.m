function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
  %SPNETCOSTSLAVE Slave cost function for simple phone net
  %   Does all the work of cost / gradient computation
  %   Returns cost broken into cross-entropy, weight norm, and prox reg
  %        components (ceCost, wCost, pCost)

  %% default values
  po = false;
  if exist('pred_only','var')
    po = pred_only;
  end;

  % some useful variables
  m = size(data, 2);

  % add ones to the data matrix (for bias units)
  data = [ones(1, m); data];

  %% reshape into network
  stack = params2stack(theta, ei);
  numHidden = numel(ei.layer_sizes) - 1;

  % cell arrays to hold data for activations, deltas, and gradients
  hAct = cell(numHidden + 1, 1);
  deltas = cell(numHidden + 1, 1);
  gradStack = cell(numHidden + 1, 1);

  %% forward prop
  % before the for loop, set X as equal to the data (input layer)
  X = data;
  for l = 1:numel(stack)
    % constructing theta from bias vector and weight matrix in the stack
    % NOTE: since we prepended ones to the data, we prepend the bias vector
    % to the weight matrix
    Theta = [stack{l}.b stack{l}.W];
    if l < numel(stack)
      % apply the activation function the hidden layers
      Z = Theta * X;
      % add bias nodes if not computing output layer
      A = [ones(1, size(Z, 2)); ei.activation_fun(Z)];
      % set X as equal to the currect hidden layer activations
      X = A;
    elseif l == numel(stack)
      % apply softmax normalization to the output layer
      Z = exp(Theta * X);
      A = bsxfun(@rdivide, Z, sum(Z));
    end
    hAct{l} = A;
  end

  pred_prob = hAct{end};
  % fprintf('predicted probabilities for the first 5 examples\n');
  % disp(pred_prob(:, 1:5));

  %% return here if only predictions desired.
  if po
    cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
    grad = [];
    return;
  end;

  %% compute cost
  % compute cross entropy cost
  num_classes = size(hAct{end}, 1);
  indicator_matrix = sparse(labels, 1:m, 1);
  ceCost = -sum(sum(indicator_matrix .* log(hAct{end})));

  % compute weight norm cost for regularization
  wCost = 0;
  for l = 1:numel(stack)
    wCost = wCost + sum(sum(stack{l}.W .^ 2));
  end

  pCost = (ei.lambda / 2) * wCost;
  cost = (ceCost / m) + pCost;

  %% compute gradients using backpropagation
  % compute deltas
  % NOTE: The index l increments backwards from the number of hidden layers +
  % the output layer. Also note that the computation of deltas for a particular
  % layer l relies on the weights W indexed by stack{l+1}. This is because
  % stack{l+1}.W refers to the weights matrix connecting layer l and l+1.
  for l = numel(stack):-1:1
    if l == numel(stack)
      deltas{l} = -(indicator_matrix .- hAct{end});
    elseif l < numel(stack)
      % ignore bias nodes
      A = hAct{l}(2:end, :);
      deltas{l} = (stack{l+1}.W' * deltas{l+1}) .* (A .* (1 .- A));
    end
  end

  % compute gradients
  % NOTE: The index l increments forward here. This is because the gradient
  % computation algorithm relies on
  for l = 1:numel(stack)
    % ignore bias nodes
    if l == 1
      A = data(2:end, :);
    else
      A = hAct{l-1}(2:end, :);
    end
    % compute partial derivatives
    gradStack{l}.W = (deltas{l} * A') ./ m + ei.lambda * stack{l}.W;
    gradStack{l}.b = sum(deltas{l}, 2) ./ m;
  end

  %% reshape gradients into vector
  [grad] = stack2params(gradStack);
end
