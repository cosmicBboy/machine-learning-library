function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = 0.01;
sigma = 0.01;
yval = yval(:);
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
pred = svmPredict(model, Xval)(:);
err = mean(double(pred ~= yval));

C_list = [0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.03 0.1 0.3 1 3 10 30];

for c = C_list
    for s = sigma_list
        disp(c)
        disp(s)
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        pred = svmPredict(model, Xval)(:);
        new_err = mean(double(pred ~= yval));
        if new_err < err,
            err = new_err;
            C = c;
            sigma = s;
        end
    end
end


% =========================================================================

end
