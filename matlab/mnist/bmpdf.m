function [p] = bmpdf(x, k, mu, w)
%BMPDF Summary of this function goes here
%   Detailed explanation goes here
%   x: d x 1 binary (0/1) data matrix 
%   k: number of cluster
%   mu: d x k
%   w: k

p = zeros(1, k); 

for j = 1:k
    p(j) = prod((mu(:, k).^x).*((1 - mu(:, k)).^(1 - x)));
end

p = sum(p.*w);

end

