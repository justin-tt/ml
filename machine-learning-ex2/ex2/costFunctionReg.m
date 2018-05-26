function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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



% J = (1/m)*(sum( -y.*(log(sigmoid(X*theta))) - (1-y).*(log(1-sigmoid(X*theta))) )) + (lambda/(2*m))*(sum(theta'*theta));
% This was wrong, not supposed to include theta(0)

thetaWithoutThetaZero = theta(2:end,1);
J = (1/m)*(sum( -y.*(log(sigmoid(X*theta))) - (1-y).*(log(1-sigmoid(X*theta))) )) + (lambda/(2*m))*(sum(thetaWithoutThetaZero'*thetaWithoutThetaZero));



% grad = [ (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,1)))); ...
% 		 (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,2)))); ...
% 		 (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,3)))); ]

% need a n * 1 vector

grad(1) = (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,1))));

n = size(theta);
for j = 2:n
	grad(j) = (1/m)*(sum((sigmoid(X*theta)-y).*(X(:,j)))) + (lambda/m)*(theta(j));

end


% =============================================================

end
