function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region.
%
%   Use mean pooling here.

poolFilter = ones(poolDim, poolDim);
% Get the vector of indices that are relevant pool rows and columns in the
% pool convolutions
poolIndex = [1: (convolvedDim / poolDim)] * poolDim - (poolDim - 1);
% here we don't have to rot90 the filter because the pool filter values
% are all ones
% convPool = conv2()
for imageNum = 1:numImages
  im = convolvedFeatures(:, :, :, imageNum);
  for filterNum = 1:numFilters
    % get image
    % get convolution
    convFeat = im(:, :, filterNum);
    % convolve the pool filter and convolved feature matrix
    poolConvAll = conv2(convFeat, poolFilter, 'valid');
    % grab only relevant pools. We wanted disjointed regions of the
    % convolved features. Then to take the mean just divide by
    % (poolDim * poolDim)
    poolConv = poolConvAll(poolIndex, poolIndex) / (poolDim * poolDim);
    pooledFeatures(:, :, filterNum, imageNum) = poolConv;
  end
end

end

