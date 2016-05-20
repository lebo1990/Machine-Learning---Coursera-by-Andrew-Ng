function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% for i = 1: K
% 	total = zeros(m, n + 1);
% 	for j = 1: m
% 		if idx(j) == i
% 			total(j, 1) = X(j, 1);
% 			total(j, 2) = X(j, 2);
% 			total(j, 3) = 1;
% 		end
% 	end
% 	centroids(i, 1) = sum(total(:, 1)) / sum(total(:, 3));
% 	centroids(i, 2) = sum(total(:, 2)) / sum(total(:, 3));
% end

for i = 1: K
	index = idx == i;
	for j = 1: n
		centroids(i, j) = sum(X(:, j) .* index) / sum(index);
	end
end





% =============================================================


end

