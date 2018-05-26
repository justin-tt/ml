function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% X has 5 examples with 4 features
% theta is a 4 length vector

% This assignment is different from the previous one, we added 1 extra theta for the prev assignment
% but for this one we just need to zero out the first theta
thetaWithoutThetaZero = theta;
thetaWithoutThetaZero(1) = 0;


% from ex2
J = (1/m)*(sum( -y.*(log(sigmoid(X*theta))) - (1-y).*(log(1-sigmoid(X*theta))) )) + (lambda/(2*m))*(sum(thetaWithoutThetaZero'*thetaWithoutThetaZero));



% grad = [ (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,1)))); ...
% 		 (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,2)))); ...
% 		 (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,3))));
% ] %should be a vector, not a scalar!!

                   %((n+1) x 1)             (5 x 1)
grad = (1/m) * X'*(sigmoid(X*theta)-y) + ((lambda/m)*thetaWithoutThetaZero);

%grad(j) = (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,j)))) + (lambda/m)*(theta(j));


% FAQ
% I get the correct results for the One-vs-All Classifier, but the submit grader doesn't give me any score.

% The submit grader uses a different test case than the one in the exercise script. It has a different number of training examples, features, and classifications.

% Be sure your code makes no assumptions about any of these parameters. Your code must work with any size of data set.





% =============================================================

grad = grad(:);

end
