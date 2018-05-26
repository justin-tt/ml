function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(X) % (12 x 2)
% size(y) % (12 x 1)
% size(theta) % (2 x 1)
% size(lambda) % (1 x 1)

% h = X * theta  (12 x 2) * (2 x 1) = (12 x 1)

h = X * theta;
thetaJ = theta(2:end,1); % remove theta0 for regularised term

J = (1/ (2*m)) * sum((h - y).*(h-y)) + (lambda/(2*m)) * sum(thetaJ.*thetaJ);



% for i th example grad = (1 x j) features
% for vectorised example, need (m x j) matrix for grad
% transpose to give (j x m)
% collapse each example (sum columns) into a single column
% to give (j x 1)

% need to get a (j x 1) matrix for gradient

thetaJ = [0; thetaJ];

% for all examples, get grad0
% grad0 = (1/m) * sum((h-y) .* X(:,1));

% grad1 = (1/m) * sum((h-y) .* X(:,2)) + (lambda/m)*(theta(2));

% grad = [grad0 grad1];
% grad = sum(grad,1);

% ^ FIX this to be less hard coded

% grad is a (1 x 2) or (2 x 1) vector
% [grad0 grad1]

% can use (1 x 12) x (12 x 2) = (1 x 2)
grad = ((1/m) * ((h-y)' * X)) .+ ((lambda/m)*thetaJ');
% FIXED ^


% =========================================================================

grad = grad(:);

end
