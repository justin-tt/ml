function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



% for each example i - find all distances || x(i) - u j || ^2 (the magnitude)
% then pick the minimum one and assign it to c(i)

% size(X) % 300 x 2
% size(centroids) % 3 x 2

% % K = 3 there are 3 centroids
% size(idx) % 300 x 1

for exampleI = 1:length(idx)
	distanceVector = [];
	%fprintf('example: %.2f\n', exampleI);
	for centroidJ = 1:K
		% what if it's more than a 2 dimensional feature set?
		distanceSquared = 0;

		for featuresN = 1:size(X,2)
			%X(exampleI,featuresN)
			%centroids(centroidJ,featuresN)
			distanceSquared = distanceSquared + (X(exampleI,featuresN) - centroids(centroidJ,featuresN))^2;
			

		end
		distanceVector(centroidJ) = distanceSquared;
		%fprintf('distance from centroid %d: %.2f\n', centroidJ, distanceSquared);
				
	end
	[minDist, minIndex] = min(distanceVector);
	idx(exampleI) = minIndex;
end

% =============================================================

end

