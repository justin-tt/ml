function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% h(x) = theta0 + theta1 x1

% to get a vector of h(xi)  // m x 1 

% we need to multiply a vector of 1s and xi == X // m x 2
% with a vector of theta0 and theta1 == theta // 2 x 1

% dimensional analysis = (mx2) * (2x1) = (mx1)

% the 1s have been added for me already into X

h = X * theta;

% cost function = h - y

J = sum((h - y).^2) / (2*m);




% =========================================================================

end
