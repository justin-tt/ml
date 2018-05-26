function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

% size(X_rec) % 50 x 2
% size(Z) % 50 x 1
% size(U) % 2 x 2
% size(K) % 1 x 1

% X_rec = U * K';

% size(X_rec(1,:)) % (1 x 1024)

for featuresN = 1:K
	% for each dimension to recover back along
	% get U_reduce
	U_reduce = U(:, 1:K); % (2 x 1) , or (n x k)
	
	% size(U_reduce) %   (1024 x 100)


	% for each example use Xapprox(m) = Ureduce . z(m)
	% (2 x 1) vector here
	for example = 1:length(Z)
		% X_approx_m = U_reduce * Z(example,:)
		% size(Z(example,:)) % (1 x 100)
		X_rec(example,:) = (Z(example,:) * U_reduce');
		% need to get (1 x 1024), we're getting (1 x 100) - need Z x U_reduce'
	end

end

% =============================================================

end
