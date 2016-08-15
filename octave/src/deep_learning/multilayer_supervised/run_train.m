% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath ../ex1;

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784; % NOTE: default value should be 784
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim]; % NOTE default value should be 256
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% this should reference a function in the current directory (or any other
% directory in PATH)
ei.activation_fun = @sigmoid;

% adjust the data if you want to train a NN on a subset of the data
data_train = data_train(1:ei.input_dim, :);
labels_train = labels_train(:);
data_test = data_test(1:ei.input_dim, :);

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.useMex = 0;
pred_only = false;

%% run training
% num_checks = 100;
% grad_check(@supervised_dnn_cost, params, num_checks, ei, data_train,...
%   labels_train);
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
  params, options, ei, data_train, labels_train, pred_only);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
