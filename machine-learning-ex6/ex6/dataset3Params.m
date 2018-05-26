function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

% predictions = svmPredict(model, Xval);
% mean(double(predictions ~= yval)) % ~= is not equal to, double is to cast the category into a double

CArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

resultArray = [];

for i = 1:length(CArray)
	for j = 1:length(sigmaArray)
		model= svmTrain(X, y, CArray(i), @(x1, x2) gaussianKernel(x1, x2, sigmaArray(j))); 
		predictions = svmPredict(model, Xval);
		
		resultArray(i,j) = mean(double(predictions ~= yval));
	end
end

resultArray % the indices of the minimum error will give you the CArray/sigmaArray
% getting indices of a matrix
% https://www.mathworks.com/matlabcentral/answers/74835-how-do-i-get-both-the-minimum-of-a-2d-matrix-and-it-s-indices
[min_val,idx]=min(resultArray(:))
[row,col]=ind2sub(size(resultArray),idx)

C = CArray(row)
sigma = sigmaArray(col)



% =========================================================================

end
