function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% from ex3, but need to add in the sum for K classes, and also use the feed forward at the same time
a1 = [ones(size(X,1),1), X];   			% (5000 x 401)

z2 = a1 * Theta1'; 						% (5000 x 401) x (401 x 25) = (5000 x 25) 
a2 = sigmoid(z2);						
a2 = [ones(size(X,1),1), a2];			% (5000 x 26)

z3 = a2 * Theta2';						% (5000 x 26) x (26 x 10) = (5000 x 10)
a3 = sigmoid(z3);						% (5000 x 10) - a3 is actually htheta(x)
%J = (1/m)*(sum( -y.*(log(sigmoid(X*theta))) - (1-y).*(log(1-sigmoid(X*theta))) )) + (lambda/(2*m))*(sum(thetaWithoutThetaZero'*thetaWithoutThetaZero));



% size(nn_params)
% size(input_layer_size)
% size(hidden_layer_size)
% size(num_labels)
% size(X)
% size(y)


%creating a matrix form of y (as 5000x10) rather than (5000x1)
yz = zeros(m, num_labels);
for example = 1:m
	yz(example,y(example)) = 1;
end


% using dot multiplication to get htheta(x) multiplied with y, and summing the K columns then the m rows
J = (1/m) * sum(sum(( (-yz.*log(a3)) - (1-yz).*log(1-a3) ),2));


% Regularised cost function
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26


% Truncating the first column
% https://www.mathworks.com/matlabcentral/answers/262488-how-to-truncate-specific-rows-and-columns-of-matrix-in-matlab?requestedDomain=www.mathworks.com

Theta1Reg = Theta1;
Theta2Reg = Theta2;

Theta1Reg(:,1) = [];
Theta2Reg(:,1) = [];

% Theta2
% Theta2Reg

J = J + (lambda/(2*m))*(sum(sum(Theta1Reg.*Theta1Reg)) + (sum(sum(Theta2Reg.*Theta2Reg))));

% Backpropgation

% Renamed labels at the top to be consistent with the "Resources" section

% calculating d3 
d3 = a3 - yz;  	% (5000 x 10)
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);		% (5000 x 10) * (10 * 25) = (5000 x 25)    Need to remove bias layer from Theta2

Delta2 = d3' * a2; % (10 x 5000) * (5000 x 26) = (10 x 26)?
Delta1 = d2' * a1; % (25 x 5000) * (5000 x 401) = (25 x 401)


% Non-regularised gradients
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Regularised gradients


%size(Theta1_grad)
%size([zeros(size(Theta1Reg,1),1), Theta1Reg])
Theta1_grad = Theta1_grad + (lambda/m)*([zeros(size(Theta1Reg,1),1), Theta1Reg]);
Theta2_grad = Theta2_grad + (lambda/m)*([zeros(size(Theta2Reg,1),1), Theta2Reg]);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
