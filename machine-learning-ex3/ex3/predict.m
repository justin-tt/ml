function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size(Theta1)	% = 25 x 401
% size(Theta2)	% = 10 x 26
% size(X)			% = 5000 x 400


% a = sigmoid(theta1 * x)
% Layer 1
% googled "concat matrix matlab" https://www.tutorialspoint.com/matlab/matlab_matrix_concatenation.htm

% adding the bias layer to make 400 features become 401 
XWithOnes = [ones(size(X,1),1), X];
a = sigmoid(XWithOnes * Theta1');
aWithOnes = [ones(size(X,1),1), a];
h = sigmoid(aWithOnes * Theta2');

% max of each row in matlab
% https://www.mathworks.com/help/symbolic/max.html
% max(h,[],2)
% index of max of each row in matlab
[c, p] = max(h,[],2);

% =========================================================================


end
