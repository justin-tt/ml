function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Updating theta0 and theta1 simultaneously every step

    % temp(i) = theta(i) - alpha * (1 / m) * sum(h(x(i) - y(i)) * x(i))


    % theta = [0:0];

    h = X * theta;

    % need to generate a dJ/dtheta array somehow

    % TODO: Improve this line to generalise to many features and many thetas...
    theta = theta - [sum((alpha/m)*(h-y)) ;sum((alpha/m)*((h - y).*X(:,2)))];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % Will be a decreasing real number
end

end
