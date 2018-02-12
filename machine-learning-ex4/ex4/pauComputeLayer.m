function [a, z] = pauComputeLayer(x, theta)
% computes a(l+1) layer give x(l) input and theta(l).

% number of samples
m = size(x, 1);

% add bias input
X = [ones(m,1), x];

z = X * theta';
a = sigmoid(z);

% function end
end
