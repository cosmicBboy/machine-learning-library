function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calculate cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
% e.g. converts (10 x 10 x 20 x 256) > (2000 x 256)
activationsPooled = reshape(activationsPooled,[],numImages);
%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);
Z = exp(bsxfun(@plus, (Wd * activationsPooled), bd));
A = bsxfun(@rdivide, Z, sum(Z));
probs = A;

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost
% create a k-by-m boolean matrix where for each column (images), the value
% should be 0 except for the row index where the index equals the true label
% for that example.
indicator_matrix = sparse(labels, 1:numImages, 1);
% then compute the cost by multiplying the indicator matrix with log probs,
% then summing over each class and each example
cost = cost - sum(sum(indicator_matrix .* log(probs))) / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.
%  Use the kron function and a matrix of ones to do this upsampling
%  quickly.
delta_softmax = -(indicator_matrix - probs);
% errors for the pooling layer

% NOTE: I made an error here by multiplying the below term by the derivative of
% the activation function (actPooled .* (1 .- actPooled)). This step should
% simply be to take the outer product of the densely connected weights and
% the softmax deltas.
%
% This is an error because the pooled activations are not relevant for
% computing the error of the pooling layer since the pooling layer is a
% different kind of transformation that the typical feedforward transformation.
delta_p = Wd' * delta_softmax;

% reshape the delta_p from
% 2D matrix -> (outputDim x outputDim x numFilters * numImages)
delta_p = reshape(delta_p, outputDim, outputDim, numFilters, numImages);

% errors for the convolved filters
% NOTE: upsampling from the pooling layer to filter layer should have the same
% dimensions as the convolved filter matrix
% (convDim x convDim * numFilters x numImages).
delta_c = zeros(size(activations));
for imageNum = 1:numImages
  for filterNum = 1:numFilters
    % select the filterNum-th pool deltas of the imageNum-th image
    dp = delta_p(:, :, filterNum, imageNum);
    dp_up = (1 / poolDim ^ 2) * kron(dp, ones(poolDim));
    % select respective activation of the convolution filter matrix
    act_k = activations(:, :, filterNum, imageNum);
    delta_c(:, :, filterNum, imageNum) = dp_up .* (act_k .* (1 - act_k));
  end
end

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

% compute the gradient for the softmax layer
% size(Wd_grad) should == size(Wd)
% size(bd) should == size(bd)
Wd_grad = (delta_softmax * activationsPooled') / numImages;
bd_grad = sum(delta_softmax, 2) / numImages;

% compute gradient for convolution layer
for imageNum = 1:numImages
  im = images(:, :, imageNum);
  for filterNum = 1:numFilters
    dc_k = delta_c(:, :, filterNum, imageNum);
    dc_k_rot = rot90(dc_k, 2);
    gc_k = conv2(im, dc_k_rot, 'valid');
    Wc_grad(:, :, filterNum) = Wc_grad(:, :, filterNum) + gc_k / numImages;
    bc_grad(filterNum) = bc_grad(filterNum) + sum(dc_k(:)) / numImages;
  end
end

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
